# Preliminary five-minute speaking script

English draft, November 8, 2026. Based on the v17 report and the updated seven-slide deck.

Target: 4 minutes 40 seconds, leaving 20 seconds before the competition limit. This is a rehearsal estimate, not a measured recording. Read the spoken paragraphs only. Do not read slide titles, timing labels, source links or backup slides aloud.

## Slide 1: Toronto Home-Price Prediction

**Target: 0:00-0:25 (25 seconds)**

How much should a Toronto home sell for? Before listing, our Off-market model has no asking price. Once listed, our List-aware model can use the seller's advertised price. These support different decisions, so we evaluate them separately, using one hundred and fifty thousand synthetic sales and no outside prediction data.

中文提示：交代两条赛道和挂牌价的含义。

## Slide 2: Official monthly and team fixed-four-month validation

**Target: 0:25-1:20 (55 seconds)**

We use two validation designs. The official design predicts December twenty twenty-five through May twenty twenty-six, refitting before each month. It evaluates ten thousand and sixty-nine homes per track. Our fixed four-month design trains before February and predicts February through May without updates, covering seven thousand seven hundred and seventy-three homes. This mirrors the final four-month forecast. Both designs train on several years of data. The ninety-, one-hundred-eighty-, and three-hundred-sixty-five-day windows are feature lookbacks, rather than training lengths. Each historical query excludes future, same-day and self-sale outcomes.

中文提示：指着两种验证设计说明：逐月重训与固定四个月预测。不要把历史特征窗口说成训练期。

## Slide 3: LightGBM and gradient-boosted trees

**Target: 1:20-2:00 (40 seconds)**

LightGBM combines decision trees. A tree follows rules about a home's features and assigns a score at a leaf. Later trees add corrections guided by the loss. LightGBM grows the leaf with the largest expected improvement and groups numeric values into bins to search efficiently. That suits nonlinear interactions between location, size and property type. We constrain tree size and apply penalties to limit complexity. Prediction combines the frozen trees.

中文提示：解释树的分支和叶子，再解释逐步加树纠正误差。

## Slide 4: From tree scores to home prices

**Target: 2:00-2:40 (40 seconds)**

Off-market predicts log sale price using property-type routes, with related types sharing training examples. Houses and townhouses use LightGBM, while apartments use CatBoost. List-aware uses one global LightGBM to predict the log sale-to-asking-price ratio. We multiply that adjustment by the asking price. Both models minimize absolute error in log space. This emphasizes relative differences and reduces the influence of extreme errors compared with squared loss. It approximates our goal, without directly optimizing median percentage error.

中文提示：区分预测房价与预测成交/挂牌比，简要说明损失函数。

## Slide 5: Results under both validation protocols

**Target: 2:40-3:40 (60 seconds)**

Under official monthly validation, Off-market achieves seven point six two percent median absolute percentage error. For List-aware, simply predicting the asking price gives seven point two six percent. Our ratio model reduces that to three point nine three percent. Under our fixed four-month design, Off-market gives seven point nine five percent, and List-aware gives three point nine three percent. Mean errors also appear in the table. These are the same selected model specifications. Different homes, cutoffs and updating rules prevent a direct ranking across protocols. Configuration selection overlapped December, so both results remain cumulative development validation. Hidden-test performance is unknown.

中文提示：先读官方结果，再读团队固定四个月结果；说明口径不同、开发期复用。无需逐项朗读 MAPE。

## Slide 6: High-price homes remain difficult

**Target: 3:40-4:15 (35 seconds)**

This figure uses our fixed four-month validation. Off-market error rises to sixteen point six seven percent in the highest-price decile, compared with six point four two in the lowest. Missing quality information or shrinkage toward typical homes could contribute, but this chart cannot establish the cause. Realized-price groups diagnose past errors and cannot directly route future predictions. Expensive or unusual homes deserve extra review.

中文提示：先强调这是固定四个月的图，再讲高价房误差和人工复核。

## Slide 7: Use of the estimates and next steps

**Target: 4:15-4:40 (25 seconds)**

Off-market supports initial valuation before listing. List-aware adjusts the asking-price anchor once a home is listed. We recommend extra review for unusual homes. Next, we would test an unused period and calibrate uncertainty intervals from past data. Model explanations describe fitted behavior, while hidden-test accuracy remains unknown. Thank you.

中文提示：给出使用方式和下一步，用一句话保留隐藏测试未知。

## Rehearsal notes

- The spoken draft contains 495 words. Timing depends on pronunciation, pauses and slide changes. Rehearse aloud and leave time before 5:00.
- Keep the distinction between percent and percentage points. The asking-price rows are baselines that directly predict the listed price.
- The fixed-four-month chart retains the original February-May cohort. Its overall Off-market label is 7.95%, which differs from the 7.62% official monthly result because the evaluation protocol differs.
- If time runs short, shorten the tree-growth explanation on slide 3. Keep the validation distinction, development-reuse disclosure and failure analysis.
- Record one full rehearsal and verify that the spoken delivery, including slide changes, finishes before 5:00.
