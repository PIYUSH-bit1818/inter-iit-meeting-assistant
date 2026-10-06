# Generates a short synthetic multi-speaker meeting recording for demos and
# end-to-end testing, using the speech voices built into Windows.
#
#   powershell -ExecutionPolicy Bypass -File scripts\make_demo_recording.ps1
#
# Output: data\demo\demo_meeting.wav (data\ is gitignored - do not commit audio).
# The meeting deliberately contains the cases the problem statement cares about:
# a confirmed decision, a negative decision, proposals that are NOT decisions,
# a question that is NOT a decision, an action item with owner + deadline, one
# with neither, statements that are NOT action items, numbers, a date, money,
# a percentage, technical terms and negations.

param([string]$OutFile = (Join-Path $PSScriptRoot "..\data\demo\demo_meeting.wav"))

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech

$outDir = Split-Path -Parent $OutFile
New-Item -ItemType Directory -Force $outDir | Out-Null
$OutFile = Join-Path (Resolve-Path $outDir) (Split-Path -Leaf $OutFile)

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$installed = $synth.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo.Name }

function Pick([string[]]$names) {
    foreach ($n in $names) { if ($installed -contains $n) { return $n } }
    return $installed[0]
}

# Three speakers; falls back to whatever voices exist on the machine.
$ananya = @{ Voice = (Pick @("Microsoft Zira Desktop", "Microsoft Zira")); Rate = 0 }
$rahul  = @{ Voice = (Pick @("Microsoft David Desktop", "Microsoft David")); Rate = 0 }
$vikram = @{ Voice = (Pick @("Microsoft Mark", "Microsoft David Desktop")); Rate = -2 }

$script = @(
    @($ananya, "Good morning everyone. This is the weekly platform sync for October 7th. Let's start with the Kubernetes upgrade."),
    @($rahul,  "Thanks Ananya. The staging cluster is still on version 1.27. I think we should go straight to version 1.29."),
    @($ananya, "Does everyone agree with upgrading to 1.29?"),
    @($vikram, "Yes, that works for me."),
    @($ananya, "Great. So we have agreed to upgrade the cluster to version 1.29 before the October 20th release."),
    @($rahul,  "The upgrade budget is 42,000 dollars, which is about 15 percent over plan."),
    @($vikram, "Maybe we could also move the nightly backups to 2 AM, but that's just an idea for now."),
    @($rahul,  "For the analytics dashboard, we could try PostgreSQL instead of MySQL."),
    @($ananya, "Let's not decide either of those today. Has everyone agreed to the new on-call rotation?"),
    @($rahul,  "Not yet. I still have concerns about the weekend shifts."),
    @($ananya, "Okay. We also decided not to change the database schema during this sprint."),
    @($ananya, "Rahul, can you update the API documentation by Friday?"),
    @($rahul,  "Sure, I'll have it done by Friday."),
    @($vikram, "I'll send the cost report to the finance team."),
    @($rahul,  "Someone should probably review the monitoring alerts at some point."),
    @($ananya, "We also need to update the onboarding guide. Thanks everyone, see you next week.")
)

$pause = New-Object System.Speech.Synthesis.PromptBuilder
$pause.AppendBreak([TimeSpan]::FromMilliseconds(600))

$synth.SetOutputToWaveFile($OutFile)
foreach ($line in $script) {
    $speaker, $text = $line
    $synth.SelectVoice($speaker.Voice)
    $synth.Rate = $speaker.Rate
    $synth.Speak($text)
    $synth.Speak($pause)
}
$synth.SetOutputToNull()
$synth.Dispose()

Write-Host "Wrote $OutFile (voices: $($ananya.Voice), $($rahul.Voice), $($vikram.Voice))"
