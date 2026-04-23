const vscode = require('vscode');

function getPanelUrl() {
  const config = vscode.workspace.getConfiguration('gjw');
  const port = config.get('port', 3456);
  return `http://localhost:${port}`;
}

class GjwWebviewProvider {
  constructor(_extensionUri) {
    this._extensionUri = _extensionUri;
    this._view = undefined;
  }

  resolveWebviewView(webviewView, _context, _token) {
    this._view = webviewView;
    webviewView.webview.options = {
      enableScripts: true,
      localResourceRoots: [this._extensionUri],
    };

    webviewView.webview.html = this._getHtml(webviewView.webview);
  }

  reload() {
    if (this._view) {
      this._view.webview.html = this._getHtml(this._view.webview);
    }
  }

  _getHtml(webview) {
    const nonce = getNonce();
    const panelUrl = getPanelUrl();
    return /*html*/ `<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy"
    content="default-src 'none'; frame-src ${panelUrl}; style-src 'unsafe-inline';">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <style>
    html, body, iframe { margin:0; padding:0; width:100%; height:100%; border:none; overflow:hidden; }
    body { background:#0d1117; }
    .loading {
      position:absolute; inset:0; display:flex; flex-direction:column;
      align-items:center; justify-content:center; gap:12px;
      color:#8b949e; font-family:system-ui,sans-serif; font-size:13px;
    }
    .loading .spinner {
      width:28px; height:28px; border:2px solid rgba(88,166,255,.2);
      border-top-color:#58a6ff; border-radius:50%;
      animation:spin .8s linear infinite;
    }
    @keyframes spin { to { transform:rotate(360deg); } }
    iframe { display:none; }
    iframe.loaded { display:block; }
  </style>
</head>
<body>
  <div class="loading" id="loading">
    <div class="spinner"></div>
    <div>正在连接gjw...</div>
    <div style="font-size:11px;color:#6e7681;">确保 server.js 已在运行</div>
  </div>
  <iframe
    id="frame"
    src="${panelUrl}"
    sandbox="allow-scripts allow-same-origin allow-forms"
    onload="document.getElementById('loading').style.display='none';this.classList.add('loaded');"
  ></iframe>
</body>
</html>`;
  }
}

function getNonce() {
  let text = '';
  const possible = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
  for (let i = 0; i < 32; i++) {
    text += possible.charAt(Math.floor(Math.random() * possible.length));
  }
  return text;
}

function activate(context) {
  const provider = new GjwWebviewProvider(context.extensionUri);

  context.subscriptions.push(
    vscode.window.registerWebviewViewProvider('gjw.chatPanel', provider, {
      webviewOptions: { retainContextWhenHidden: true },
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('gjw.openPanel', () => {
      vscode.commands.executeCommand('gjwPanel.focus');
    })
  );

  context.subscriptions.push(
    vscode.commands.registerCommand('gjw.reloadPanel', () => {
      provider.reload();
    })
  );
}

function deactivate() {}

module.exports = { activate, deactivate };
