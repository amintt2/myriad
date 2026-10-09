; Inno Setup script of the Myriad installer (per-user install, no administrator rights needed).
;   iscc /DAppVersion=0.2.0 /DSourceDir=..\..\dist\Myriad /DOutputDir=..\..\dist packaging\windows\myriad.iss
; Code signing: pass /DSignTool=1 and define a sign tool named "myriad" in ISCC (iscc "/Smyriad=signtool sign
; /fd sha256 /tr http://timestamp.digicert.com /td sha256 /f cert.pfx /p PASSWORD $f"); see release.yml.

#ifndef AppVersion
  #define AppVersion "0.2.0"
#endif
#ifndef SourceDir
  #define SourceDir "..\..\dist\Myriad"
#endif
#ifndef OutputDir
  #define OutputDir "..\..\dist"
#endif

[Setup]
AppId={{6E3C1B7A-4F2D-4C0B-9A5E-2D7F8B1C9E40}
AppName=Myriad
AppVersion={#AppVersion}
AppVerName=Myriad {#AppVersion}
AppPublisher=Myriad contributors
AppPublisherURL=https://github.com/amintt2/myriad
AppSupportURL=https://github.com/amintt2/myriad/issues
AppUpdatesURL=https://github.com/amintt2/myriad/releases
DefaultDirName={localappdata}\Programs\Myriad
DefaultGroupName=Myriad
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
OutputDir={#OutputDir}
OutputBaseFilename=Myriad-Setup-{#AppVersion}
SetupIconFile=..\icons\myriad.ico
UninstallDisplayIcon={app}\Myriad.exe
UninstallDisplayName=Myriad
Compression=lzma2/ultra64
SolidCompression=yes
WizardStyle=modern
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
LicenseFile=..\..\..\LICENSE
; Updates from the app (myriad/updater.py) run this installer with
;   /SILENT /SUPPRESSMSGBOXES /NORESTART /CLOSEAPPLICATIONS [/MYRIADRELAUNCH=1]
; after Myriad has stopped its node and llama-server. A Myriad still exiting is closed through the
; Restart Manager (the filter includes the bundle's .pyd modules); Myriad is not restarted by it but by
; the [Run] entry below, only when the app asked for a relaunch.
CloseApplications=yes
CloseApplicationsFilter=*.exe,*.dll,*.pyd
RestartApplications=no
#ifdef SignTool
SignTool=myriad
SignedUninstaller=yes
#endif

[Languages]
Name: "french"; MessagesFile: "compiler:Languages\French.isl"
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked
Name: "autostart"; Description: "{cm:AutoStartProgram,Myriad}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\Myriad"; Filename: "{app}\Myriad.exe"
Name: "{autodesktop}\Myriad"; Filename: "{app}\Myriad.exe"; Tasks: desktopicon
Name: "{userstartup}\Myriad"; Filename: "{app}\Myriad.exe"; Parameters: "--hidden"; Tasks: autostart

[Run]
Filename: "{app}\Myriad.exe"; Description: "{cm:LaunchProgram,Myriad}"; Flags: nowait postinstall skipifsilent
; Silent update started by the app ("Update and restart" button): start the new version again.
; --after-update makes it wait for the old instance to release its lock; /MYRIADHOME= gives back the
; data directory the old instance used (--home).
Filename: "{app}\Myriad.exe"; Parameters: "{code:MyriadRelaunchParams}"; Flags: nowait runasoriginaluser; Check: MyriadRelaunch

[Code]
function MyriadRelaunch(): Boolean;
begin
  Result := WizardSilent() and (ExpandConstant('{param:MYRIADRELAUNCH|0}') = '1');
end;

function MyriadRelaunchParams(Param: String): String;
var
  Home: String;
begin
  Result := '--after-update';
  Home := ExpandConstant('{param:MYRIADHOME|}');
  if (Home <> '') and (Pos('"', Home) = 0) then
    Result := Result + ' --home "' + Home + '"';
end;

[UninstallRun]
; Stop a running instance (and its llama-server) before removing the files.
Filename: "{app}\Myriad.exe"; Parameters: "--stop"; Flags: runhidden waituntilterminated; RunOnceId: "StopMyriad"

; The user's data (key, configuration, models) stays in %APPDATA%\myriad, and in %APPDATA%\essaim (the data
; directory of the versions before the rename, copied to %APPDATA%\myriad on first start): the uninstaller
; deletes neither.
