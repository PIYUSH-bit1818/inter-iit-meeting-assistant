# Generates a synthetic three-person engineering meeting for demos and
# end-to-end testing, using the speech voices built into Windows plus FFmpeg.
#
#   powershell -ExecutionPolicy Bypass -File scripts\make_demo_recording.ps1
#   powershell -ExecutionPolicy Bypass -File scripts\make_demo_recording.ps1 -OutFile samples\demo_meeting\meeting.wav
#
# Requires: Windows (System.Speech) and ffmpeg on PATH.
# Default output: data\demo\demo_meeting.wav (data\ is gitignored).
#
# The script is fictional. It deliberately contains what the problem
# statement cares about: two confirmed decisions (one negative), proposals
# that are NOT decisions, a question that is NOT a decision, an action item
# with owner + deadline, one with neither, statements that are NOT action
# items, technical terms (Kubernetes, PostgreSQL, API, CI/CD), a version
# number, a date, a monetary amount and negations.

param([string]$OutFile = (Join-Path $PSScriptRoot "..\data\demo\demo_meeting.wav"))

$ErrorActionPreference = "Stop"
Add-Type -AssemblyName System.Speech
if (-not (Get-Command ffmpeg -ErrorAction SilentlyContinue)) { throw "ffmpeg is required on PATH" }

$outDir = Split-Path -Parent $OutFile
New-Item -ItemType Directory -Force $outDir | Out-Null
$OutFile = Join-Path (Resolve-Path $outDir) (Split-Path -Leaf $OutFile)

$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$installed = $synth.GetInstalledVoices() | Where-Object { $_.Enabled } | ForEach-Object { $_.VoiceInfo.Name }
function Pick([string[]]$names) {
    foreach ($n in $names) { if ($installed -contains $n) { return $n } }
    return $installed[0]
}
$female = Pick @("Microsoft Zira Desktop", "Microsoft Zira")
$male = Pick @("Microsoft David Desktop", "Microsoft David")

# Three speakers. Arjun uses the male voice pitched down ~11% so he sounds
# clearly different from Rahul even on machines with only two voices.
$speakers = @{
    Rahul = @{ Voice = $male;   Rate = 0;  Pitch = 1.0 }
    Priya = @{ Voice = $female; Rate = 0;  Pitch = 1.0 }
    Arjun = @{ Voice = $male;   Rate = -1; Pitch = 0.89 }
}

# Each entry: speaker, text, pause after (ms)
$meeting = @(
    @("Rahul", "Good morning, everyone. Thanks for joining. Let's get started with the platform sync for this week.", 700),
    @("Priya", "Morning, Rahul.", 400),
    @("Arjun", "Good morning.", 600),
    @("Rahul", "First item on the agenda. Priya, can you give us a quick update?", 500),
    @("Priya", "Sure. Kubernetes is the main one. Our staging cluster is still on version 1.27, and support for that version ends soon.", 600),
    @("Arjun", "Right. I tested version 1.29 on a separate cluster last week. The deployment went through without problems, and our API services came up cleanly.", 600),
    @("Priya", "Did you see any issues with the CI/CD pipeline?", 450),
    @("Arjun", "Only a small one. Two of the build jobs needed an updated image, but that's already fixed.", 700),
    @("Rahul", "Okay. Based on the discussion, are we agreed that we'll migrate the Kubernetes cluster to version 1.29?", 450),
    @("Priya", "Yes, I agree.", 350),
    @("Arjun", "Yes, that works for me.", 450),
    @("Rahul", "Great. That's our decision then. We will migrate the Kubernetes cluster to version 1.29.", 800),
    @("Priya", "Next, the database. I know some people wanted to rename a few tables, but I don't think that's a good idea right now.", 500),
    @("Arjun", "I agree. Changing the schema in the middle of a migration feels risky.", 450),
    @("Rahul", "So, to be clear, we have decided that we will not change the database schema during this sprint. Everyone okay with that?", 400),
    @("Priya", "Yes.", 300),
    @("Arjun", "Agreed.", 800),
    @("Arjun", "While we're on the database, should we also consider moving the reporting database to PostgreSQL?", 500),
    @("Priya", "Maybe, but I don't think that's necessary right now. Let's discuss that next week.", 500),
    @("Rahul", "Yes, let's leave that open for now.", 700),
    @("Priya", "Another idea. Maybe we could schedule the nightly backup at 2 AM instead of midnight, when traffic is lower.", 500),
    @("Rahul", "Could be. Let's look at the traffic numbers first and come back to it next week.", 700),
    @("Arjun", "On that note, we need to think about improving the backup process in general.", 450),
    @("Priya", "And someone should probably review the monitoring alerts at some point. We get a lot of noise at night.", 700),
    @("Rahul", "Fair points. Let's talk about the budget. The current budget for the migration is forty-two thousand dollars.", 500),
    @("Arjun", "That should be enough, as long as we don't add new nodes before the release.", 500),
    @("Priya", "We'll send the updated cost report to the finance team.", 700),
    @("Rahul", "Good. One more question. Has everyone agreed to the new deployment schedule?", 450),
    @("Arjun", "Not yet. I still have some concerns about the weekend releases.", 450),
    @("Rahul", "Okay, then we'll keep that one open.", 700),
    @("Priya", "Rahul, please update the deployment documentation by Friday, October 10.", 400),
    @("Rahul", "Sure, I'll do that.", 600),
    @("Rahul", "I think that covers everything. Thanks, everyone. Talk to you next week.", 400),
    @("Priya", "Thanks, bye.", 300),
    @("Arjun", "Bye.", 300)
)

$work = Join-Path ([System.IO.Path]::GetTempPath()) ("demo_meeting_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force $work | Out-Null
$rate = 22050
$list = @()
try {
    for ($i = 0; $i -lt $meeting.Count; $i++) {
        $who, $text, $pauseMs = $meeting[$i]
        $s = $speakers[$who]
        $rawLine = Join-Path $work ("line_{0:D3}_raw.wav" -f $i)
        $synth.SelectVoice($s.Voice)
        $synth.Rate = $s.Rate
        $synth.SetOutputToWaveFile($rawLine)
        $synth.Speak($text)
        $synth.SetOutputToNull()

        # Trim the engine's own leading/trailing silence (the scripted pause is
        # added separately), resample every clip to the same format, and
        # pitch-shift where configured
        $line = Join-Path $work ("line_{0:D3}.wav" -f $i)
        $trim = "silenceremove=start_periods=1:start_threshold=-50dB,areverse," +
                "silenceremove=start_periods=1:start_threshold=-50dB,areverse"
        if ($s.Pitch -ne 1.0) {
            $filter = "$trim,asetrate={0}*{1},aresample={0},atempo={2}" -f $rate, $s.Pitch, (1.0 / $s.Pitch)
        } else {
            $filter = "$trim,aresample=$rate"
        }
        & ffmpeg -nostdin -v error -y -i $rawLine -af $filter -ac 1 -ar $rate -c:a pcm_s16le $line
        if ($LASTEXITCODE -ne 0) { throw "ffmpeg failed on line $i" }

        $gap = Join-Path $work ("gap_{0:D3}.wav" -f $i)
        & ffmpeg -nostdin -v error -y -f lavfi -t ($pauseMs / 1000.0) -i "anullsrc=r=${rate}:cl=mono" -c:a pcm_s16le $gap
        $list += "file '$($line -replace '\\', '/')'"
        $list += "file '$($gap -replace '\\', '/')'"
    }
    $listFile = Join-Path $work "list.txt"
    [System.IO.File]::WriteAllLines($listFile, $list)
    # -3 dB keeps peaks well clear of clipping
    & ffmpeg -nostdin -v error -y -f concat -safe 0 -i $listFile -af "volume=-3dB" -ac 1 -ar $rate -c:a pcm_s16le $OutFile
    if ($LASTEXITCODE -ne 0) { throw "ffmpeg concat failed" }
}
finally {
    $synth.Dispose()
    Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
}

Write-Host "Wrote $OutFile (Rahul: $male, Priya: $female, Arjun: $male pitched down)"
