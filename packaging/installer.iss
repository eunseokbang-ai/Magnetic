; Inno Setup script for the drone magnetic survey processing program.
;
; Build (after packaging/dist/Magnetic exists - see magnetic.spec):
;   "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" packaging\installer.iss
;
; Produces packaging\output\MagneticSetup.exe - one file to hand over.
; The target machine needs neither Python nor Node.js.
;
; Installs per user (no administrator rights) into the user's own
; AppData\Local\Programs, deliberately: the program writes its basemap
; tile cache next to itself, which a standard account cannot do under
; Program Files, and requiring an admin prompt for a survey tool is
; friction the people it is handed to should not have to clear.

#define AppName "드론 자력탐사 자료처리 프로그램"
#define AppShortName "Magnetic"
#define AppVersion "1.0"
#define AppPublisher "Magnetic Survey"

[Setup]
AppId={{8F3C2A91-5B47-4E2D-9A16-7C4E8D0B2F35}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={localappdata}\Programs\{#AppShortName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
OutputDir=output
OutputBaseFilename=MagneticSetup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\Magnetic.exe

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
Source: "dist\Magnetic\Magnetic.exe"; DestDir: "{app}"; Flags: ignoreversion
Source: "dist\Magnetic\_internal\*"; DestDir: "{app}\_internal"; Flags: ignoreversion recursesubdirs createallsubdirs
Source: "..\docs\Magnetic_Processing_Technical_Manual.pdf"; DestDir: "{app}\docs"; Flags: ignoreversion skipifsourcedoesntexist

[Icons]
Name: "{group}\{#AppName}"; Filename: "{app}\Magnetic.exe"
Name: "{group}\기술 매뉴얼"; Filename: "{app}\docs\Magnetic_Processing_Technical_Manual.pdf"
Name: "{group}\{#AppName} 제거"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\Magnetic.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\Magnetic.exe"; Description: "지금 실행"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; The basemap tile cache the program downloads while it runs; it is
; regenerated on demand, so leaving it behind would just be litter.
Type: filesandordirs; Name: "{localappdata}\MagneticSurvey"
