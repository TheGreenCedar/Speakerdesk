; 0.7.7's updater passes no scope flag. Keep that upgrade in its per-user scope.
!define MULTIUSER_INSTALLMODE_DEFAULT_CURRENTUSER
!define MUI_DIRECTORYPAGE_TEXT_TOP "Choose a writable folder. For Program Files, select 'Anyone who uses this computer' on the install-mode page and run setup as administrator. For a per-user installation, choose a folder in your user profile."

; Check the nearest existing parent without creating anything while choosing a folder.
Function .onVerifyInstDir
  Push $0
  Push $1
  StrCpy $0 $INSTDIR
  ${Do}
    IfFileExists "$0\*.*" parent_found
    IfFileExists "$0" invalid_directory
    ${GetParent} "$0" $1
    ${If} $1 == ""
    ${OrIf} $1 == $0
      Goto invalid_directory
    ${EndIf}
    StrCpy $0 $1
  ${Loop}
  parent_found:
  ; Need files and subdirectories. Program Files permits these for administrators
  ; without DELETE_CHILD on its root; requiring that would reject valid installs.
  System::Call 'kernel32::CreateFileW(w r0, i 0x6, i 7, p 0, i 3, i 0x02200000, p 0) p.r1'
  ${If} $1 == -1
    Goto invalid_directory
  ${EndIf}
  System::Call 'kernel32::CloseHandle(p r1)'
  Pop $1
  Pop $0
  Return
  invalid_directory:
  Pop $1
  Pop $0
  Abort
FunctionEnd

!macro NSIS_HOOK_PREINSTALL
  ; Passive/silent installs skip the directory page. Fail before copying or registration.
  ClearErrors
  ; NSIS clears an invalid /D destination. Do not let the probe fall back to root.
  ${If} $INSTDIR == ""
    SetErrors
  ${Else}
    GetTempFileName $0 $INSTDIR
  ${EndIf}
  ${If} ${Errors}
    SetErrorLevel 5
    MessageBox MB_OK|MB_ICONSTOP "Speakerdesk cannot write to this folder. Choose a folder in your user profile, or run setup as administrator and select 'Anyone who uses this computer' for Program Files." /SD IDOK
    Abort
  ${EndIf}
  Delete $0

  ${If} $MultiUser.InstallMode == "CurrentUser"
    System::Call 'shlwapi::PathIsPrefixW(w "$PROGRAMFILES64", w "$INSTDIR") i.r0'
    System::Call 'shlwapi::PathIsPrefixW(w "$PROGRAMFILES32", w "$INSTDIR") i.r1'
    ${If} $0 <> 0
    ${OrIf} $1 <> 0
      SetErrorLevel 5
      MessageBox MB_OK|MB_ICONSTOP "For Program Files, go back and select 'Anyone who uses this computer'. For a per-user installation, choose a folder in your user profile." /SD IDOK
      Abort
    ${EndIf}
  ${EndIf}
!macroend

!macro NSIS_HOOK_PREUNINSTALL
  ; Both scopes may coexist. Bind cleanup to this uninstaller's own directory.
  ReadRegStr $0 HKCU "${UNINSTKEY}" "InstallLocation"
  ReadRegStr $1 HKLM "${UNINSTKEY}" "InstallLocation"
  ${If} $0 == '$\"$INSTDIR$\"'
  ${AndIf} $1 != '$\"$INSTDIR$\"'
    SetShellVarContext current
  ${ElseIf} $1 == '$\"$INSTDIR$\"'
  ${AndIf} $0 != '$\"$INSTDIR$\"'
    SetShellVarContext all
  ${Else}
    SetErrorLevel 2
    MessageBox MB_OK|MB_ICONSTOP "Speakerdesk's installation registration does not identify this folder uniquely. Reinstall Speakerdesk in the same folder before uninstalling." /SD IDOK
    Abort
  ${EndIf}
!macroend
