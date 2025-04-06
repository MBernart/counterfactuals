## Analyesd papers
- 1 [Analysis of Explainers of Black Box Deep Neural Networks for Computer Vision: A Survey](https://arxiv.org/pdf/1911.12116)
- 2 [Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead](https://arxiv.org/pdf/1811.10154)
- 3 [Interpreting Black-Box Models: A Review on Explainable Artificial Intelligence](https://link.springer.com/article/10.1007/s12559-023-10179-8)
- 4 [COUNTERFACTUAL EXPLANATIONS WITHOUTOPENING THE BLACK BOX: AUTOMATED DECISIONS AND THE GDPR](https://arxiv.org/pdf/1711.00399)
- 5 https://github.com/SeldonIO/alibi

	## Some ideas for improvements regarding main paper
- change ideal point to minimize impact of outliers
$$
z_i = median \{ g_i(x) : x ∈ D \}
$$  
- sampling around ideal point to improve slightly (exploit good solution)
- ...

## Ideas regarding above papers
- Forest Regressor for selecting features [for some samples]
- Certifiably Optimal Rule Lists (CORELS) for features [For black box?!]
- *H-statistic* for sampled space
- LIME (Local Interpretable Model-Agnostic Explanations) for simplifying and exploiting local areas - explain locally
- Clustering sampled counter class then moving forward -> moving from ideal point to ideal point of diff class
- use loss function to exploit ?close? scenarios between some different classes samples [4] between two classes like 
$$
X^′ =arg min_{X^′ } d(X,X^′ )+loss(X^′ )
$$
^Go with above with prof. Szczęch^