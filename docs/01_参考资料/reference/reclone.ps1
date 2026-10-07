# reclone.ps1 — reference/（第⑥层 第三方开源参考层）一键重建脚本
#
# 用途：按锁定清单 docs/01_参考资料/reference/reference-lock.md 把 15 家第三方开源参考库
#       重新克隆到目标目录，每家 clone 后 checkout 到**锁定提交号**，保证任何一台机器重建出
#       完全一致的只读参考层。
# 项目规则（AGENTS.md 五）：reference/ 是"第三方开源项目参考层"，只读阅读用（读源码查实现/查规范），
#       **不进 git**（.gitignore 已忽略 reference/），**不 pip install 进 .venv**（装包需另行请示用户）。
#       本脚本只做"下载源码目录 + 钉住提交号"，仅此而已。
#
# 用法：
#   .\reclone.ps1                                   # 克隆到 <项目根>\reference
#   .\reclone.ps1 -TargetRoot D:\tmp\reference      # 克隆到指定目录
#   .\reclone.ps1 -Only tdxpy,bt                    # 只重建指定几家（补克隆/实测用）
#
# 深度说明：machine-learning-for-trading 与 vectorbt 两家上游体积超 300MB，
#       按约定用 --depth 1 浅克隆（见 reference-lock.md 备注②）。浅克隆只含最新快照，
#       重建时先 `git fetch --depth 1 origin <锁定SHA>` 再 checkout（GitHub 支持按 SHA 浅取，已实测）。
# 失败处理：任何一步 git 失败立即抛错终止（无静默 fallback）；已存在的目标目录拒绝覆盖。
#
# 锁定日期：2026-10-07。修改锁定提交号前先更新 reference-lock.md，两处必须一致。

[CmdletBinding()]
param(
    [string]$TargetRoot,
    [string[]]$Only = @()
)

$ErrorActionPreference = 'Stop'

if (-not $TargetRoot) {
    # 默认 <项目根>\reference —— 本脚本位于 docs/01_参考资料/reference/，向上三级即项目根（不硬编码绝对路径）
    $TargetRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..\..\..\reference'))
}

function Invoke-Git {
    param(
        [string]$WorkDir,
        [string[]]$GitArgs,
        [string]$What
    )
    if ($WorkDir) { & git -C $WorkDir @GitArgs } else { & git @GitArgs }
    if ($LASTEXITCODE -ne 0) {
        throw "$What 失败（git exit=$LASTEXITCODE）：git $($GitArgs -join ' ')"
    }
}

# name | url | 锁定提交号 | shallow（上游 >300MB 才用 --depth 1）
$table = @(
    @{ name = 'qlib';                        url = 'https://github.com/microsoft/qlib.git';                        commit = 'be725493eb1a6bbb42bf11b37aa7669f59610ff1'; shallow = $false }
    @{ name = 'alphalens-reloaded';          url = 'https://github.com/stefan-jansen/alphalens-reloaded.git';       commit = 'f0a07c22d554e4b4036983cc80320b432714fe7e'; shallow = $false }
    @{ name = 'vectorbt';                    url = 'https://github.com/polakowo/vectorbt.git';                     commit = 'ceffc501f2d37033a79dd86a9f883e69ec6977bd'; shallow = $true  }
    @{ name = 'rqalpha';                     url = 'https://github.com/ricequant/rqalpha.git';                     commit = '0d98adefa87956e26f3e7ca5b26b5f3fc7ca834f'; shallow = $false }
    @{ name = 'bt';                          url = 'https://github.com/pmorissette/bt.git';                        commit = '1ce8e84e95b6c055cdd2765c4f1b7a2fe0bb5efe'; shallow = $false }
    @{ name = 'OpenAlpha';                   url = 'https://github.com/ziyouqitan/OpenAlpha.git';                  commit = '214b0e3cdab11fa8fb1949724d0543baff333625'; shallow = $false }
    @{ name = 'mlfinlab';                    url = 'https://github.com/hudson-and-thames/mlfinlab.git';            commit = '79dcc7120ec84110578f75b025a75850eb72fc73'; shallow = $false }
    @{ name = 'machine-learning-for-trading'; url = 'https://github.com/stefan-jansen/machine-learning-for-trading.git'; commit = '6790e76326b6cab2117bb80abd6b07345bf0762d'; shallow = $true }
    @{ name = 'adata';                       url = 'https://github.com/1nchaos/adata.git';                         commit = 'b14f4e57b2175302f18b6eaf934f7dff9207a141'; shallow = $false }
    @{ name = 'efinance';                    url = 'https://github.com/Micro-sheep/efinance.git';                  commit = 'c8fd370a3109b2d14a121e3a32a86e9c8354b01b'; shallow = $false }
    @{ name = 'mootdx';                      url = 'https://github.com/mootdx/mootdx.git';                         commit = 'e99ae34382d970c68654c6d17c45512e728f130d'; shallow = $false }
    @{ name = 'eltdx';                       url = 'https://github.com/electkismet/eltdx.git';                     commit = '59d4615d3dafed4ecbdc480a34f96807cb132b92'; shallow = $false }
    @{ name = 'easyquotation';               url = 'https://github.com/shidenggui/easyquotation.git';              commit = '7778c1b9f93afb2ce2cf431de393fe690d4ec07c'; shallow = $false }
    @{ name = 'tdxpy';                       url = 'https://github.com/One-sixth/tdxpy.git';                       commit = '37acfccc14d758b92030444c7039ae300c146ca0'; shallow = $false }
    @{ name = 'Ashare';                      url = 'https://github.com/mpquant/Ashare.git';                        commit = '7ef1ce07579416d5c9e58f713867132bbaa9390d'; shallow = $false }
)

New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null
Write-Host "目标目录：$TargetRoot"

$planned = @($table | Where-Object { $Only.Count -eq 0 -or $Only -contains $_.name })
if ($planned.Count -eq 0) { throw "-Only 未匹配到任何项目：$($Only -join ',')" }
Write-Host ("计划重建 {0} 家" -f $planned.Count)

$i = 0
foreach ($r in $planned) {
    $i++
    $dst = Join-Path $TargetRoot $r.name
    Write-Host ("[{0}/{1}] {2}  <-  {3}" -f $i, $planned.Count, $r.name, $r.url)
    if (Test-Path $dst) { throw "目标已存在，拒绝覆盖：$dst（先人工确认删除或换 -TargetRoot）" }

    if ($r.shallow) {
        Invoke-Git -GitArgs @('clone', '--depth', '1', $r.url, $dst) -What "克隆 $($r.name)"
        # 浅克隆只有最新快照，tip 可能已前移 —— 按锁定 SHA 浅取再 checkout
        Invoke-Git -WorkDir $dst -GitArgs @('fetch', '--depth', '1', 'origin', $r.commit) -What "浅取 $($r.name) 锁定提交"
    }
    else {
        Invoke-Git -GitArgs @('clone', $r.url, $dst) -What "克隆 $($r.name)"
    }

    Invoke-Git -WorkDir $dst -GitArgs @('checkout', $r.commit) -What "checkout $($r.name) 锁定提交"

    $head = (git -C $dst rev-parse HEAD 2>&1 | Out-String).Trim()
    if ($head -ne $r.commit) {
        throw "checkout 后 HEAD 与锁定提交号不一致：$($r.name) 实际=$head 锁定=$($r.commit)"
    }
    Write-Host ("    OK {0} @ {1}" -f $r.name, $head)
}

Write-Host ""
Write-Host "== 重建结果核对 =="
foreach ($r in $planned) {
    $dst = Join-Path $TargetRoot $r.name
    $head = (git -C $dst rev-parse HEAD 2>&1 | Out-String).Trim()
    Write-Host ("{0}  {1}  match={2}" -f $r.name, $head, ($head -eq $r.commit))
}
Write-Host "完成。reference/ 不进 git，未做任何 git add/commit。"