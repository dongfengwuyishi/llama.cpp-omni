; Inno Setup script for Comni (MiniCPM-o 4.5 Windows desktop)
; Produces a single-file installer: Comni-Setup-<version>-win64.exe
;
; Build:
;   ISCC.exe Comni.iss
; Or via the one-click script:
;   powershell -File make_installer.ps1

#define MyAppName        "Comni"
#define MyAppPublisher   "OpenBMB"
#define MyAppURL         "https://github.com/OpenBMB/MiniCPM-o"
#define MyAppExeName     "Comni.exe"
#define MyAppDescription "MiniCPM-o 4.5 Desktop (Windows)"

; MyAppVersion and MySourceDir can be overridden from the command line:
;   ISCC.exe /DMyAppVersion=1.0.0 /DMySourceDir=..\..\..\..\dist\Comni Comni.iss
#ifndef MyAppVersion
  #define MyAppVersion "1.0.0"
#endif
#ifndef MySourceDir
  #define MySourceDir "..\..\..\..\dist\Comni"
#endif
#ifndef MyOutputDir
  #define MyOutputDir "..\..\..\..\release"
#endif

[Setup]
AppId={{A6B3F7F8-0C92-4F3B-91D9-COMNI-MINICPM-O}}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppVerName={#MyAppName} {#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
AppSupportURL={#MyAppURL}
AppUpdatesURL={#MyAppURL}
VersionInfoVersion={#MyAppVersion}
VersionInfoProductName={#MyAppName}
VersionInfoDescription={#MyAppDescription}

; Install by default into the user's Local AppData to avoid UAC prompts and
; keep the embedded Python + model cache writable by the user.
DefaultDirName={localappdata}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
DisableDirPage=no
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; x64 only — our llama-server.exe and CUDA runtime are 64-bit.
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
MinVersion=10.0

; Compression. LZMA2/max is slow to build but yields the smallest installer.
Compression=lzma2/max
SolidCompression=yes
LZMAUseSeparateProcess=yes
LZMANumBlockThreads=4

; Output
OutputDir={#MyOutputDir}
OutputBaseFilename={#MyAppName}-Setup-{#MyAppVersion}-win64
SetupIconFile=Comni.ico
UninstallDisplayIcon={app}\{#MyAppExeName}
UninstallDisplayName={#MyAppName}
WizardStyle=modern

; Close running instance gracefully on reinstall/uninstall.
CloseApplications=yes
RestartApplications=no

[Languages]
; ChineseSimplified.isl is an unofficial translation bundled next to this
; .iss file (not shipped by default with Inno Setup 6).
Name: "chinesesimplified"; MessagesFile: "ChineseSimplified.isl"
Name: "english";           MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon";   Description: "{cm:CreateDesktopIcon}";         GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "startmenuicon"; Description: "Create a Start Menu shortcut";   GroupDescription: "{cm:AdditionalIcons}"

[Files]
Source: "{#MySourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}";                     Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"; Tasks: startmenuicon
Name: "{group}\{cm:UninstallProgram,{#MyAppName}}"; Filename: "{uninstallexe}"
Name: "{userdesktop}\{#MyAppName}";                Filename: "{app}\{#MyAppExeName}"; IconFilename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#MyAppName}}"; Flags: nowait postinstall skipifsilent

[UninstallDelete]
; Logs / caches written next to the executable (just in case the app stored any).
; User models in %USERPROFILE%\.comni\ and logs in %APPDATA%\Comni\ are intentionally kept.
Type: filesandordirs; Name: "{app}\_internal\resources\apps\server\logs"
