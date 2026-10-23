package engine

import (
	"archive/zip"
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"testing/fstest"
)

func TestParseNvidiaSMI(t *testing.T) {
	g := ParseNvidiaSMI("NVIDIA GeForce RTX 4090, 24564, 560.94\r\nNVIDIA GeForce RTX 3060, 12288, 560.94\r\n")
	if len(g) != 2 || g[0].Name != "NVIDIA GeForce RTX 4090" || g[0].MemoryMB != 24564 || g[1].Driver != "560.94" {
		t.Fatalf("%+v", g)
	}
	if len(ParseNvidiaSMI("")) != 0 {
		t.Fatal("empty output gave gpus")
	}
}

func TestRecommendVariant(t *testing.T) {
	cases := []struct {
		gpus []GPU
		want string
	}{
		{nil, "cpu"},
		{[]GPU{{Name: "NVIDIA GeForce RTX 5090", Driver: "576.02"}}, "cu128"},
		{[]GPU{{Name: "NVIDIA GeForce RTX 3060", Driver: "560.94"}}, "cu126"},
		{[]GPU{{Name: "NVIDIA GeForce GTX 1060", Driver: "472.12"}}, "cu118"},
	}
	for _, c := range cases {
		if got := RecommendVariant(c.gpus); got != c.want {
			t.Errorf("%v: got %s want %s", c.gpus, got, c.want)
		}
	}
	for _, v := range Variants {
		if !validVariant(v.ID) {
			t.Fatal(v.ID)
		}
	}
	if validVariant("cu999") {
		t.Fatal("bogus variant accepted")
	}
}

func TestEnableSite(t *testing.T) {
	dir := t.TempDir()
	pth := filepath.Join(dir, "python312._pth")
	os.WriteFile(pth, []byte("python312.zip\r\n.\r\n\r\n# Uncomment to run site.main() automatically\r\n#import site\r\n"), 0o644)
	if err := EnableSite(dir); err != nil {
		t.Fatal(err)
	}
	if err := EnableSite(dir); err != nil { // idempotent
		t.Fatal(err)
	}
	b, _ := os.ReadFile(pth)
	s := string(b)
	if strings.Count(s, "import site") != 1 || strings.Contains(s, "#import site") {
		t.Fatalf("site: %q", s)
	}
	if strings.Count(s, `Lib\site-packages`) != 1 || !strings.Contains(s, "python312.zip") {
		t.Fatalf("lib: %q", s)
	}
	if err := EnableSite(t.TempDir()); err == nil {
		t.Fatal("missing pth should error")
	}
}

func TestUnzipRejectsTraversal(t *testing.T) {
	dir := t.TempDir()
	mk := func(name string, entries ...string) string {
		p := filepath.Join(dir, name)
		f, _ := os.Create(p)
		zw := zip.NewWriter(f)
		for _, e := range entries {
			w, _ := zw.Create(e)
			w.Write([]byte("x"))
		}
		zw.Close()
		f.Close()
		return p
	}
	good := mk("good.zip", "python.exe", "Lib/os.py")
	out := filepath.Join(dir, "out")
	if err := Unzip(good, out); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(out, "Lib", "os.py")); err != nil {
		t.Fatal(err)
	}
	bad := mk("bad.zip", "../evil.txt")
	if err := Unzip(bad, filepath.Join(dir, "out2")); err == nil {
		t.Fatal("traversal accepted")
	}
}

func TestDownload(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/missing" {
			http.NotFound(w, r)
			return
		}
		w.Write([]byte(strings.Repeat("a", 1<<20)))
	}))
	defer srv.Close()
	dst := filepath.Join(t.TempDir(), "f")
	var last int64
	if err := Download(context.Background(), srv.Client(), srv.URL+"/f", dst, func(d, _ int64) { last = d }); err != nil {
		t.Fatal(err)
	}
	if fi, _ := os.Stat(dst); fi.Size() != 1<<20 || last != 1<<20 {
		t.Fatalf("size %d last %d", fi.Size(), last)
	}
	if err := Download(context.Background(), srv.Client(), srv.URL+"/missing", dst+"2", nil); err == nil {
		t.Fatal("404 accepted")
	}
	if _, err := os.Stat(dst + "2.part"); err == nil {
		t.Fatal("partial left behind")
	}
}

func TestSyncApp(t *testing.T) {
	p := NewPaths(t.TempDir())
	files := fstest.MapFS{
		"airllm_worker.py":     {Data: []byte("print(1)")},
		"airllm/__init__.py":   {Data: []byte("")},
		"airllm/__pycache__/x": {Data: []byte("junk")},
	}
	script, err := SyncApp(p, files)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(p.App, "airllm", "__init__.py")); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(p.App, "airllm", "__pycache__")); err == nil {
		t.Fatal("pycache copied")
	}
	// unchanged content: no rewrite (a marker file survives)
	marker := filepath.Join(p.App, "marker")
	os.WriteFile(marker, nil, 0o644)
	if _, err := SyncApp(p, files); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(marker); err != nil {
		t.Fatal("rewrote unchanged app")
	}
	files["airllm_worker.py"] = &fstest.MapFile{Data: []byte("print(2)")}
	if _, err := SyncApp(p, files); err != nil {
		t.Fatal(err)
	}
	b, _ := os.ReadFile(script)
	if string(b) != "print(2)" {
		t.Fatal("changed app not rewritten")
	}
	if _, err := os.Stat(marker); err == nil {
		t.Fatal("stale files kept")
	}
}

func TestEnv(t *testing.T) {
	e := strings.Join(Env(true, `D:\models`), "\n")
	if !strings.Contains(e, "AIRLLM_WORKER_USE_BUNDLED=1") || !strings.Contains(e, `HF_HOME=D:\models`) || !strings.Contains(e, "PYTHONUTF8=1") {
		t.Fatal(e)
	}
	if strings.Contains(strings.Join(Env(false, ""), "\n"), "HF_HOME") {
		t.Fatal("empty cache set HF_HOME")
	}
}
