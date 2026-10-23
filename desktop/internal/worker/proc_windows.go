//go:build windows

package worker

import (
	"os/exec"
	"syscall"
)

const createNoWindow = 0x08000000

// hideWindow keeps the child python.exe from flashing a console window.
func hideWindow(cmd *exec.Cmd) {
	cmd.SysProcAttr = &syscall.SysProcAttr{HideWindow: true, CreationFlags: createNoWindow}
}

// HideWindow is the exported form for other packages that spawn processes (pip, nvidia-smi).
func HideWindow(cmd *exec.Cmd) { hideWindow(cmd) }
