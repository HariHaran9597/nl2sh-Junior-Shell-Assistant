const vscode = require('vscode');
const { execFile } = require('child_process');
function activate(context){
  let d = vscode.commands.registerCommand('nl2sh.translate', async ()=>{
    const nl = await vscode.window.showInputBox({prompt: 'Describe the shell command in plain English'});
    if(!nl) return;
    const editor = vscode.window.activeTextEditor;
    execFile('python', ['-m', 'cli.nl2sh', '-q', nl], {cwd: vscode.workspace.workspaceFolders?.[0]?.uri.fsPath || undefined}, (err,stdout)=>{
      if(err){ vscode.window.showErrorMessage(err.message); return; }
      const cmd = stdout.trim();
      if(editor){ editor.edit(e=> e.insert(editor.selection.active, cmd)); }
      else { vscode.env.clipboard.writeText(cmd); vscode.window.showInformationMessage(`nl2sh: ${cmd} (copied)`); }
    });
  });
  context.subscriptions.push(d);
}
function deactivate(){}
module.exports={activate,deactivate};
