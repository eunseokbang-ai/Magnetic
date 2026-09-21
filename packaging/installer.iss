; Inno Setup script for DroneMag Studio.
;
; Build (after packaging\dist\DroneMagStudio exists - see magnetic.spec):
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
;
; Produces packaging\output\DroneMagStudioSetup.exe - one file to hand
; over. The target machine needs neither Python nor Node.js.
;
; Installs per user (no administrator rights) into the user's own
; AppData\Local\Programs, deliberately: the program writes its basemap
; tile cache and its autosaves under the user profile, and requiring an
; admin prompt for a survey tool is friction the people it is handed to
; should not have to clear.

#define AppName "DroneMag Studio"
#define AppNameKo "DroneMag Studio - 드론 자력탐사 자료처리"
#define AppShortName "DroneMagStudio"
#define AppVersion "1.1"
#define AppPublisher "DroneMag Studio"

[Setup]
AppId={{8F3C2A91-5B47-4E2D-9A16-7C4E8D0B2F35}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppNameKo} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\{#AppShortName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=output
OutputBaseFilename=DroneMagStudioSetup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
; What the Windows "Apps & features" list and the uninstall entry show.
UninstallDisplayName={#AppNameKo} {#AppVersion}
UninstallDisplayIcon={app}\{#AppShortName}.exe

; Installation password. Inno asks for this before any file is written.
; Encryption=yes also encrypts the payload inside the setup file, so the
; program cannot simply be read out of the installer without it.
;
; Worth being plain about what this is and is not: it keeps the program
; from being installed by someone who was not given the password, but it
; is not protection against a determined reader - anyone who installs it
; once has the files, and the password itself lives in this script. Treat
; it as a gate, not as licensing.
Password=0428683019
Encryption=yes

[Languages]
Name: "korean"; MessagesFile: "compiler:Languages\Korean.isl"

[Tasks]
Name: "desktopicon"; Description: "바탕화면에 바로가기 만들기"; GroupDescription: "추가 아이콘:"

[Files]
Source: "dist\{#AppShortName}\{#AppShortName}.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\{#AppShortName}\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\docs\Magnetic_Processing_Technical_Manual.pdf"; DestDir: "{app}\docs"; DestName: "DroneMagStudio_기술매뉴얼.pdf"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\{#AppShortName}.exe"
Name: "{group}\{#AppName} 기술 매뉴얼"; Filename: "{app}\docs\DroneMagStudio_기술매뉴얼.pdf"
Name: "{group}\{#AppName} 제거"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppShortName}.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppShortName}.exe"; Description: "지금 실행"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; The basemap tile cache the program downloads while it runs; it is
; regenerated on demand, so leaving it behind would just be litter. The
; autosaved projects live in the same tree and are the user's own work,
; so they are kept - see the [Code] section, which asks first.
Type: filesandordirs; Name: "{localappdata}\DroneMagStudio\tiles"
Type: filesandordirs; Name: "{localappdata}\DroneMagStudio\intermagnet_cache"

[Code]
// Autosaved projects are work, not cache, so uninstalling does not throw
// them away silently. Asked once, and only when there is something there.
procedure CurUninstallStepChanged(CurUninstallStep: TUninstallStep);
var
  AutosaveDir: String;
begin
  if CurUninstallStep = usPostUninstall then
  begin
    // A silent uninstall has nobody to answer, and this prompt would
    // simply hang it (measured: /VERYSILENT sat waiting on this dialog).
    // Keeping the work is the safe answer when nobody is asked.
    if UninstallSilent then
      Exit;
    AutosaveDir := ExpandConstant('{localappdata}\DroneMagStudio\autosave');
    if DirExists(AutosaveDir) then
    begin
      if MsgBox('자동 저장된 프로젝트가 남아 있습니다.' #13#10 +
                AutosaveDir + #13#10#13#10 +
                '함께 삭제할까요? (다시 설치해서 이어서 작업하려면 [아니오])',
                mbConfirmation, MB_YESNO) = IDYES then
        DelTree(AutosaveDir, True, True, True);
    end;
  end;
end;
