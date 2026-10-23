package main

import (
	"embed"
	"io/fs"
	"log"

	"github.com/wailsapp/wails/v2"
	"github.com/wailsapp/wails/v2/pkg/options"
	"github.com/wailsapp/wails/v2/pkg/options/assetserver"
	"github.com/wailsapp/wails/v2/pkg/options/windows"
)

//go:embed all:frontend/dist
var assets embed.FS

// bundle holds airllm_worker.py and a copy of the repo's airllm package; build.sh fills it.
//
//go:embed all:bundle
var bundle embed.FS

func main() {
	files, err := fs.Sub(bundle, "bundle")
	if err != nil {
		log.Fatal(err)
	}
	app := NewApp(files)

	err = wails.Run(&options.App{
		Title:            "AirLLM",
		Width:            1280,
		Height:           820,
		MinWidth:         960,
		MinHeight:        620,
		Frameless:        true,
		BackgroundColour: &options.RGBA{R: 14, G: 15, B: 17, A: 255},
		AssetServer:      &assetserver.Options{Assets: assets},
		OnStartup:        app.startup,
		OnShutdown:       app.shutdown,
		Bind:             []interface{}{app},
		SingleInstanceLock: &options.SingleInstanceLock{
			UniqueId:               "b1e2a7c4-airllm-desktop",
			OnSecondInstanceLaunch: app.secondInstance,
		},
		Windows: &windows.Options{
			Theme:                windows.Dark,
			WebviewIsTransparent: false,
			WindowIsTranslucent:  false,
			DisablePinchZoom:     true,
		},
	})
	if err != nil {
		log.Fatal(err)
	}
}
