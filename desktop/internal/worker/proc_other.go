//go:build !windows

package worker

import "os/exec"

func hideWindow(cmd *exec.Cmd) {}

func HideWindow(cmd *exec.Cmd) { hideWindow(cmd) }
