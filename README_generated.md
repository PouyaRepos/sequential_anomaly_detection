# Sequential Learning for Optimal Detection of Rare Events

## Table of Contents
- [Introduction](#introduction)
- [Al Realizable Case](#al-realizable-case)
- [Al Noisy Case](#al-noisy-case)

---

## Introduction

The prompt and accurate detection of rare events is a critical challenge across various real-world domains due to the potentially severe consequences of such events. Examples include financial market crashes, power grid failures, cybersecurity breaches, early detection of rare diseases, and issues encountered by autonomous systems. Rare events are defined by their low probability of occurrence; however, when they do occur, their impacts can be substantial. In the financial sector, for instance, early detection of market anomalies can avert significant economic losses. Similarly, timely identification of early symptoms of rare diseases in healthcare can be life-saving. In autonomous systems, accurately differentiating between a real pedestrian and a reflection on a surface is paramount. The rarity and high-stakes nature of these events necessitate the development of detection methods that maximize detection probabilities while minimizing false positives and associated costs.

Detecting rare events poses significant challenges, primarily due to their infrequency, the inherent imbalance in the data, and the difficulty in distinguishing between normal and anomalous patterns. Traditional detection methods often rely on static models, which passively analyze data and utilize fixed thresholds to identify anomalies. These approaches are frequently inadequate for rare event detection, as they are unable to adapt to new information in real time and often require extensive labeled datasets—acquisition of which is both costly and impractical.

To address these limitations, this thesis proposes a framework that combines active learning with sequential decision-making, referred to as Sequential Learning for Optimal Detection of Rare Events. This approach integrates active learning strategies with dynamic decision-making to optimize resource allocation and improve detection performance. Sequential learning frameworks are inherently adaptive, enabling continuous model updates based on newly acquired data while allowing for the strategic selection of the most informative data points for labeling. Active learning functions as a feedback-driven system, wherein the algorithm iteratively requests and incorporates new information to refine its model. This adaptive approach is particularly well-suited for rare event detection, where labeled data is sparse and expensive to obtain.

The feedback loop intrinsic to active learning frameworks also facilitates the reconstruction of decision boundaries and dynamic adjustment of instance labels as new information becomes available. This dynamic adaptability is especially valuable in scenarios where patterns evolve or where unanticipated rare events emerge [hino2020active].

Throughout this thesis, the terms active learning and sequential learning are used interchangeably, underscoring the iterative and adaptive nature of the methodology in addressing the complexities of rare event detection.



## Motivation

The primary focus of this thesis is the optimal detection of rare events using active learning strategies. In this context, it is essential to precisely define the terms *Active Learning*, *Optimal Detection*, and *Rare Events*, both conceptually and mathematically, to establish a rigorous foundation for the methodologies employed.

*Active Learning*: This research emphasizes active learning, where the algorithm selectively labels the most informative data points rather than passively processing large volumes of potentially irrelevant data. This selective approach is particularly advantageous for rare event detection, as it reduces the cost and effort associated with labeling while prioritizing data points that contribute most to model improvement. Unlike passive learning, which assumes all data points are equally informative and typically selects them randomly, active learning strategically identifies instances with the potential to enhance model accuracy, especially for rare events that might otherwise be overlooked.

*Optimal Detection*: The objective of optimal detection in this context is to devise a strategy that maximizes detection accuracy while minimizing detection delays and false positives. This requires balancing exploration (acquiring new information) and exploitation (leveraging existing knowledge to detect rare events) to efficiently navigate the data space. Given the infrequency of these events, the learning algorithm must explore the data space effectively to identify rare occurrences with minimal resource expenditure, including computational resources and labeling costs.

*Rare Events*: Rare events are instances that, due to their low probability and infrequent occurrence, constitute a minority within the dataset. Despite their rarity, their potential severity necessitates careful monitoring and accurate detection to prevent significant consequences. Examples from domains such as finance, healthcare, and autonomous systems underscore the critical nature of these events. The definition of a rare event is inherently context-dependent, determined by the distribution characteristics and frequency within the dataset. Furthermore, the nature of rare events may evolve; an event initially classified as rare could become more common, necessitating the model's adaptability in detection strategies.

In many machine learning tasks, obtaining large quantities of unlabeled data is relatively straightforward and cost-effective. However, acquiring labeled data remains a significant challenge, often requiring substantial investments of time, effort, or specialized expertise. This is largely due to the need for manual annotation or expert input to produce meaningful labels, making the process labor-intensive and resource-demanding. Consequently, a pronounced disparity exists between the abundance of unlabeled data and the difficulty of acquiring labeled data, which is a common bottleneck in the development and training of machine learning models [hanneke2014theory].

For example, training models for self-driving cars necessitates millions of hours of street traffic videos. A mere ten-second video can comprise approximately 300 frames, each requiring the labeling of multiple objects, which is an extremely time-consuming process. Specifically, one second of video consists of around 30 frames, and if each frame contains 20 cars that need to be boxed (labeled), annotating just 10 seconds of video could take nearly 100 minutes, as illustrated in Figure \ref{Figs/video frame.png}. This exemplifies a critical bottleneck in machine learning: while the acquisition of data is often straightforward, the labeling process is both arduous and resource-intensive.  



To address this challenge, Active/Sequential Learning has emerged as a highly effective paradigm. This approach enables models to focus on the most informative instances, thereby reducing the volume of labeled data required and significantly enhancing efficiency. Unlike passive learning, which relies on randomly selected labeled datasets and lacks feedback between the model and the data selection process, active learning strategically prioritizes data points near decision boundaries. These points are the most likely to improve the model’s performance, particularly in complex scenarios.

This thesis aims to develop an active learning framework capable of identifying the most informative data points for labeling, particularly in the context of rare and high-impact events. By addressing this challenge, the thesis contributes to the development of cost-effective and accurate detection strategies that can adapt to real-world data. Furthermore, it emphasizes the comparative analysis of realizable and noisy cases to enhance robustness in practical applications.

The realizable and noisy cases will be discussed in greater technical detail in Chapters \ref{chapter: Realizable} and \ref{chapter: Noisy}, respectively. As a precursor to this discussion, it is valuable to highlight their general properties, as well as the key similarities and differences between these cases.


## Realizable and Noisy Cases
The purpose of this work is to compare the performance of realizable and noisy cases in the detection of rare events. The realizable case is employed as a benchmark, as it assumes ideal conditions that shield the model from noise and distractions. While such assumptions rarely align with real-world scenarios, they provide a controlled setting to evaluate and measure the efficiency and behavior of alternative methods, such as those designed for noisy cases. 

The realizable case offers a foundational framework, enabling the exploration of noisy cases, which are more representative of real-world conditions. In noisy cases, data collection, labeling, and model fitting are often influenced by numerous unpredictable factors, creating a more challenging and realistic environment. The following sections provide a non-technical overview of these two scenarios.


### Realizable Case
Most active learning approaches are based on the assumption that data are perfectly separable, the hypothesis space contains a function capable of perfectly classifying both training and testing data, and there is no error or noise in the labeling process [kaariainen2006active]. However, these assumptions are often unrealistic in real-world scenarios.

The first widely recognized general-purpose approach for realizable cases was introduced by Cohn, Atlas, and Ladner [cohn1994improving], commonly referred to as the CAL algorithm, named after its creators. The CAL strategy operates as follows: the algorithm sequentially evaluates each instance within the unlabeled data pool. If two classifiers exist in the version space (a subset of classifiers/hypothesis class consistent with all previously observed labels) that disagree on the label of the current instance, the algorithm requests the oracle to provide the label. Otherwise, no label is requested. Meanwhile, inconsistent classifiers are removed from the version space. 

Algorithms inspired by CAL are categorized as disagreement-based methods. These methods are sometimes referred to as "mellow" active learning because they adopt a minimalistic approach: the algorithm only queries labels when it cannot infer the label from the available information, without actively seeking out the most informative examples. In this framework, informativeness is treated as a binary characteristic—an instance is either deemed informative or not, with no further differentiation based on degrees of informativeness [hanneke2012activized]. Notably, the algorithm does not account for the extent of disagreement among classifiers. For example, it does not differentiate between an instance that is inconsistent with a single classifier and one for which the majority of classifiers disagree. Additionally, the sequential querying process and the removal of incompatible classifiers significantly influence the data selection for labeling.


### Noisy Case
A realistic scenario arises when the assumptions of the realizable case are relaxed. Significant progress has been made in addressing the challenges posed by imperfect annotators, underspecified feature spaces, and model specification errors [hanneke2012activized]. In general, noisy cases—whether in passive or active learning—are considerably more challenging than realizable cases.

This domain of active learning is commonly referred to as *agnostic active learning*, which has its roots in the agnostic PAC (Probably Approximately Correct) model [kearns1992toward]. In this setting, a wide variety of classifiers, such as linear models, neural networks, and decision trees, can be employed. However, no flawless classifier exists, and in some instances, labels may also be noisy. The primary objective in such cases is to identify a classifier that minimizes the discrepancy with the best possible hypothesis.

A notable advancement in agnostic active learning was achieved with the introduction of the `A^2` algorithm by Balcan, Beygelzimer, and Langford [balcan2006agnostic, balcan2009agnostic]. Designed to be resilient to noise, this algorithm utilizes disagreement-based strategies and is applicable to various classifiers under diverse noise conditions [hanneke2012activized]. The core assumption underpinning their work is that the `A^2` algorithm operates with access to a continuous supply of unlabeled data points sampled independently and identically from a stationary distribution (i.i.d.). However, this assumption does not hold in many real-world scenarios. For instance, data drifting is a common occurrence, particularly when sensors are repaired or replaced, leading to non-stationary data distributions.

Active learning inherently seeks to select samples based on their informativeness rather than randomly. Thus, while the overall unlabeled dataset may adhere to the i.i.d. assumption, the specific data points chosen for labeling—limited by a budget—often deviate from this assumption.

Furthermore, the output of their work is restricted to a narrow class of classifiers and is, in some sense, parametric. An alternative line of research seeks to alleviate this restriction by adopting a *nonparametric* setting, which aims to expand the relationship between parametric distributions and learning rates [locatelli2017adaptivity]. Minsker [minsker2012plug], along with Locatelli, Carpentier, and Kpotufe [locatelli2017adaptivity], have addressed this issue under the conventional nonparametric assumptions for regression functions. Specifically, the regression function is assumed to belong to a specified Hölder class `\Sigma(\beta, K, [0,1]^k)` and to satisfy the low-noise condition.

This work introduces a novel approach based on *plug-in classifiers*. This method demonstrates superior performance compared to passive algorithms, particularly under the *Tsybakov low-noise* condition. This condition is characterized by the existence of positive constants `B` and `\lambda > 0` such that:

```
    \forall t > 0, \textnormal{P}_{X} (x: \left|\eta(x) \right| \leq t) \leq Bt^{\lambda}, 
```
where ` \mathcal{X}\in\mathbb{R}^{k}` and `\mathcal{Y}\in\{0, 1\}` are couple from unknown `\textnormal{P}_{X,Y}` distribution, with `\textnormal{P}_{X}` as the marginal distribution of `\mathcal{X}`. And
`\eta(x) = \mathbb{E}(Y|X=x)`, a regression function from Hölder class `\Sigma(\beta,K, [0,1]^k)`, where `(X,Y)\sim \textnormal{P}_{X,Y}`.


The related algorithm achieves exponential convergence in *label complexity*, a significant improvement over the typically polynomial rates of passive learning. The plug-in classifier discussed in this context refers to a classification method based on the nonparametric estimation of the regression function, `\eta(x)`, which belongs to the Hölder class `\Sigma(\beta, K, [0,1]^k)`. The central idea of a plug-in classifier is to estimate this regression function and utilize the sign of the estimate to make predictions. The plug-in classifier introduced in [minsker2012plug] operates as follows:

In a binary classification problem where the labels `Y` are either 0 or 1, the regression function `\eta(x)` provides the expected value of `Y` given a feature vector `X = x`. Specifically, `\eta(x)` represents the probability that a label is 1, with higher values indicating that `Y` is more likely to be 1. The classifier's objective is to predict `Y` using `X`, which is achieved by determining whether `\eta(x)` is greater than or less than zero.

Since the true regression function `\eta(x)` is unknown, it is estimated from the data using a nonparametric method, such as local polynomial regression. Once `\eta(x)` is estimated (denoted as `\hat{\eta}(x)`), the plug-in classifier is defined as `f(x) = \textnormal{sign}(\hat{\eta}(x))`. This function `f(x)` classifies a data point `x` based on the sign of the estimated regression function. The term "plug-in" classifier arises from the fact that the decision boundary is determined by substituting the estimate `\hat{\eta}(x)` into the decision rule ` \textnormal{sign}(\hat{\eta}(x))`.

The classifier employs a piecewise-constant estimator for `\textnormal{sign}(\hat{\eta}(x))`. The feature space is partitioned into small regions (dyadic cubes), within which the regression function is assumed to be constant. For each region `R_i` in the feature space, the estimator `\textnormal{sign}(\hat{\eta}(x))` assigns the average of the observed labels for all data points within that region.

The algorithm presented in [locatelli2017adaptivity], which will be explored in this work, builds upon the approach in [minsker2012plug] with some notable differences. It operates as follows:

The algorithm begins with a hierarchical dyadic partitioning of the feature space `[0,1]^k`. The space is divided into dyadic cells, which are cubes of progressively smaller sizes as the algorithm refines the grid. These cells collectively cover the entire domain of interest. The purpose of this partitioning is to focus the classification task on regions of interest, particularly near the decision boundary where classification uncertainty is highest. As the algorithm progresses, the diameter of the cells decreases, enabling increasingly fine local estimates of `\eta(x)`.

For each cell in the partition, the algorithm computes a local estimate of `\eta(x)`, which represents the probability that the label is 1. This estimate is typically based on the labels of points within the cell or in neighboring cells. The key idea is to prioritize regions where `\eta(x)` is close to 0.5, as these regions indicate uncertainty in classification—whether to label `x` as 0 or 1. These areas are critical for labeling since small changes in `\eta(x)` can significantly impact classification decisions. 

This step is central to active learning as it enables the algorithm to focus on the most informative points, thereby reducing label complexity. Once the regions of uncertainty are identified, the algorithm requests labels for points within these regions. The strategy is to concentrate sampling efforts in areas where classification is most challenging, allowing the algorithm to reduce uncertainty and refine the classifier effectively.

The algorithm iteratively refines the partition and requests labels until a predefined stopping condition is met. This stopping condition may depend on the labeling budget, the degree of uncertainty reduction achieved, or a specified confidence level in the classifier's performance.


### Version Space
In Active Learning, a key concept is the *version space*, which plays a critical role in guiding the selection of informative samples. The version space is defined as the subset of the hypothesis space that contains all classifiers consistent with the labeled data observed so far. This concept is particularly prominent in realizable active learning, where the hypothesis class is assumed to be fixed and finite. In this context, the version space progressively narrows as the algorithm receives more labeled instances, enabling the learner to strategically query data points that are most likely to reduce the version space. The objective is to eliminate as many inconsistent classifiers as possible, thereby converging to the optimal classifier with minimal labeling effort.

In contrast, noisy active learning does not assume a predefined hypothesis class. Instead, the version space starts undefined, and the algorithm incrementally learns a model from the data without assuming a fixed parameterization. Rather than explicitly managing a version space, noisy active learning represents classifiers through data-driven approaches such as sampling-based methods or kernel-based estimators. Techniques like nearest neighbors, decision trees, or Gaussian processes are commonly employed. As labeled data accumulates, the algorithm implicitly updates its classifier representation, dynamically adjusting the decision boundary without explicitly maintaining a well-defined version space.

The distinction between realizable and noisy cases also influences how uncertainty is measured and how the algorithm selects subsequent queries. In realizable settings, uncertainty is often derived from disagreements among the classifiers within the version space. Conversely, in noisy settings, uncertainty typically stems from data density or model variance and is addressed using methods like uncertainty sampling or expected model change strategies. While both frameworks aim to minimize labeling effort while ensuring high accuracy, the underlying mechanisms for representing and updating classifiers differ fundamentally between realizable and noisy cases.


### Label Complexity
Label complexity reflects the number of labeled data points required for a learning algorithm to achieve a specified level of accuracy. In both realizable and noisy cases, label complexity is influenced by various factors, though the key parameters differ between the two.

According to [hanneke2014theory], in the *realizable case*, label complexity is determined by several critical factors. First, it depends on the *VC dimension* (`d`) of the hypothesis class, which quantifies its complexity. A higher VC dimension typically increases label complexity, as more labeled data is needed to narrow down the set of consistent hypotheses. Another essential factor is the *target error* (`\varepsilon`); achieving higher accuracy (lower `\varepsilon`) necessitates more labels, with label complexity growing logarithmically as `1/\varepsilon`. Additionally, the *disagreement coefficient* (`\theta`) plays a pivotal role by quantifying the likelihood that a new data point lies in the region of disagreement among consistent hypotheses. A smaller disagreement coefficient reduces label complexity, as it indicates faster shrinkage of the version space with each query.

In contrast, as described in [locatelli2017adaptivity], the *noisy case* introduces additional complexities. Label complexity in this setting is influenced by the *noise margin* (`\beta`), which measures the level of uncertainty near the decision boundary. A larger `\beta` corresponds to fewer ambiguous points, thereby lowering the label complexity. Another critical parameter is the *smoothness* of the decision boundary, represented by `\alpha`. A smoother decision boundary (larger `\alpha`) reduces label complexity, while a rougher boundary (smaller `\alpha`) increases it. The *dimensionality* (`k`) of the feature space also impacts label complexity, as higher dimensions generally require more labeled data to adequately explore the data space. Similar to the realizable case, the target error `\varepsilon` remains a factor; however, its relationship with label complexity is more intricate under noisy conditions.

This comparison highlights the distinct ways in which label complexity is influenced by the structure of the hypothesis space, noise, and data characteristics in realizable and noisy cases. While the realizable case benefits from the simplicity of assuming a perfect classifier, the noisy case introduces additional challenges due to the necessity of managing uncertainty and adapting to noise.

## Al Realizable Case

This section will investigate the active learning framework under the *realizable* case. As mentioned before the realizable case contains assumptions that do not align with real-world scenarios. Due to numerous uncontrollable factors, it is almost impossible to get a clean data set, flawless oracle, or perfect model. However, to have a benchmark to evaluate other methods against the realizable states, studying and characterizing them is substantial.

In the following first the initial setting and fundamental definitions will be provided. Then the main algorithm and related supporting theorems will be explored.


## Setting 

In this thesis, the following formal framework which is according to [hanneke2014theory] applies. Let \(\mathcal{X} \in[0, 1]^k \) be the instance space, equipped with a \(\sigma\)-algebra \(\mathcal{B}_\mathcal{X}\). For simplicity, assume \((\mathcal{X}, \mathcal{B}_{\mathcal{X})}\) is a standard Borel space. The label space is \(\mathcal{Y} = \{0, 1\}\), and the product space \(\mathcal{X} \times \mathcal{Y}\) is equipped with the product \(\sigma\)-algebra \(\mathcal{B} = \mathcal{B}_{\mathcal{X}} \bigotimes 2^{\mathcal{Y}}\). Let \(\textnormal{P}_{X,Y}\) denote a probability measure on \(\mathcal{X} \times \mathcal{Y}\), referred to as the target distribution, and let \(\textnormal{P}_{X}\) represent the marginal distribution of \(\textnormal{P}_{X,Y}\) over \(\mathcal{X}\). For each \(x \in \mathcal{X}\), define 

\[\eta(x) = \textnormal{P}_{X,Y}(Y = +1 | X = x),
\]
where \((X, Y) \sim \textnormal{P}_{X,Y}\). A function \(h: \mathcal{X} \to \mathcal{Y}\) is called a classifier, and the *error rate* of any classifier \(h\) is given by:
\[
er(h) = \textnormal{P}_{X,Y}((x, y): h(x) \neq y),
\]
which represents the probability that \(h\) makes a misclassification for a random point \((X, Y) \sim \textnormal{P}_{X,Y}\).

In this context, our focus is on learning from data, to find a classifier \(h\) that minimizes \(er(h)\) using samples drawn from \(\textnormal{P}_{X,Y}\). Let \(\mathcal{Z} = \{(X_i, Y_i)\}_{i=1}^{\infty}\) be a sequence of independent random variables distributed according to \(\textnormal{P}_{X,Y}\), referred to as the labeled data sequence. For \(m \in \mathbb{N}\), let \(\mathcal{Z}_m = \{(X_i, Y_i)\}_{i=1}^{m}\) denote the first \(m\) labeled data points, and define \(\mathcal{Z}_\mathbf{X} = \{X_i\}_{i=1}^{\infty}\) as the unlabeled data sequence. In practice, the number of unlabeled data points is large but finite; however, for analysis purposes, we assume access to the entire sequence \(\mathcal{Z}_\mathbf{X}\). The number of unlabeled samples used by the algorithm will be addressed in the respective analyses.

In the active learning protocol, the algorithm is provided with a budget \(n\) and access to the unlabeled sequence \(\mathcal{Z}_\mathbf{X}\). For `t = \{0, 1,..., n\}`, the index `i_t \in \mathbb{N}` is related to the `t`-th requested label. The algorithm can select an index \(i_1\) and request the label \(Y_{i_1}\). After receiving \(Y_{i_1}\), it can request another label \(Y_{i_2}\), and so on. This continues until the budget of \(n\) labels is exhausted, at which point the algorithm outputs a classifier \(\hat{h}\). Formally, this protocol involves a family of estimators that map the labeled data to a classifier \(\hat{h}\), with the condition that \(\hat{h}\) is conditionally independent of the labeled sequence, given the unlabeled data and the sequence \((i_1, Y_{i_1}), \dots, (i_n, Y_{i_n})\). 

 % Passive learning algorithms, on the other hand, are defined as functions that map labeled data points \(\mathcal{L} \in \bigcup_{n \in \mathbb{N}}(\mathcal{X} \times \mathcal{Y})^n\) to a classifier \(\hat{h}\). Our primary interest is in understanding how the behavior of these algorithms changes as a function of the number of labeled points, \(n\), namely, `\mathcal{A}(\mathcal{Z}_n)`.



## Definitions

In the core of active learning, there is a concept called *label complexity*. Following is its formal definition:
\begin{definition} [{[hanneke2014theory]}]
     For any active learning algorithm `\mathcal{A}`, we say `\mathcal{A}` attains a label complexity of `\Lambda` if, for any `\varepsilon \geq 0 ` and `\delta \in [0, 1]`, any distribution `\textnormal{P}_{X,Y}` over `\mathcal{X} \times \mathcal{Y}`, and for all integer `n \geq \Lambda(\varepsilon, \delta, \textnormal{P}_{X,Y}) `, the classifier `h` returned by `\mathcal{A}` after using a budget of `n` will, with probability at least `1- \delta`, satisfy the condition that the error `er(h)` is at most `\varepsilon`.  
 
\end{definition}
In active learning, the *label complexity* is a crucial metric. This is a concept based on that, it is possible to monitor the performance of the algorithm and classifiers. It represents the number of labeled examples that an active learning algorithm needs to achieve a certain level of accuracy or performance. This metric is crucial because active learning aims to minimize the number of labeled samples required, thereby reducing labeling costs. 


The label complexity of a machine learning task is influenced by several factors, including the underlying data distribution and the complexity of the hypothesis space. Highly imbalanced datasets or those with clusters of similar instances may require fewer labels to accurately model the target function. Conversely, a complex hypothesis space, characterized by a large number of potential hypotheses, may necessitate more labels to effectively generalize from the training data. Additionally, noise in the data can increase label complexity, as the learner may need to acquire additional information to distinguish true patterns from random fluctuations.



Generally speaking, for a given `\textnormal{P}_{X,Y}` distribution and maximum labeling budget `n`, we want a classier `\hat{h}` that with the probability of `1-\delta`, is not more than `\varepsilon` far from the perfect classifier. Simply, the minimum number of labels required to achieve a desired level of accuracy. As [hanneke2014theory] elucidates, we will focus on how many labels an active learning algorithm needs to achieve a low error rate compared to the best possible error rate within a specific hypothesis class `\mathcal{C}`. Specifically, having ` \nu = inf_{ h \in \mathcal{C}}er(h)`, known as the *noise rate*, we are concerned about the value of `\Lambda( \nu +\varepsilon, \delta, \textnormal{P}_{X,Y})`,  which is a function of ` \nu, \varepsilon, \delta,` and `\textnormal{P}_{X,Y}`.


In this work, in Chapter \ref{chapter: Realizable} we will investigate cases in which the value of `\nu` will be equal to zero. Having `\nu = 0` implies that the perfect classifier `f^{*} \in \mathcal{C}` is not contaminated by noises, namely `er(f^{*})= 0`.


Based on the seminal work of Vapnik and Chervonenkis [vapnik2015uniform], we define the notion of shattering. For a *set of classifiers* `\mathcal{H}`, or interchangeably *hypostasises class of* `\mathcal{H}`, a sequence of points `(x_1, ..., x_m) \in \mathcal{X}^m` is said to be shattered by `\mathcal{H}` if, for any possible labeling `(y_1, ..., y_m) \in \mathcal{Y}^m`, there exists a classifier `h \in \mathcal{H}` that correctly classifies all points in the sequence. In essence, shattering implies that `\mathcal{H}` can realize all `2^m` distinct labelings of `(x_1, ..., x_m)`. This means that the hypothesis class has enough flexibility to fit any combination of labels for that specific set of points.

The Vapnik-Chervonenkis `(VC)` dimension of a non-empty set `\mathcal{H}`, represented as `\textnormal{vc}(\mathcal{H})`, is defined as the maximum integer `m` for which there exists a set `S \in \mathcal{X}^m ` that can be shattered by `\mathcal{H}` into every possible labeling. If no such integer exists, the `VC` dimension is considered infinite. We denote `d= \textnormal{vc}(\mathcal{C}) ` and suppose `d` finite throughout the thesis. The `VC` dimension tells us the "shattering capacity" of `\mathcal{H}`— i.e., how many points it can consistently label in every possible way. The `VC` dimension helps estimate a model's generalization capacity. If a model can shatter a large number of points (high `VC` dimension), it may overfit, meaning it might fit the training data well but perform poorly on new data. Conversely, a model with a low `VC` dimension may underfit, meaning it might not have enough complexity to capture the patterns in the data.



We also have the following notations; For any set \( A \), let \( \mathbf{1}_A \) denote the indicator function for \( A \). This function assigns a value of 1 if \( x \in A \) and 0 otherwise, represented by \( \mathbf{1}_A(x) = 1 \) if \( x \in A \), and \( \mathbf{1}_A(x) = 0 \) if \( x \notin A \). Similarly, for any logical condition \( L \), we may use the notation \( \mathbf{1}[L] \), where \( \mathbf{1}[L] = 1 \) when \( L \) is true, and \( \mathbf{1}[L] = 0 \) when \( L \) is false.

For a classifier \( h \) and a set of labeled data points \( \mathcal{L} \in \bigcup_{m \in \mathbb{N}}(\mathcal{X} \times \mathcal{Y})^m \), we define the *empirical error rate* of \( h \) on \( \mathcal{L} \) as 
\[
\text{er}_{\mathcal{L}}(h) = \frac{1}{|\mathcal{L}|} \sum_{(x, y) \in \mathcal{L}} \mathbf{1}[h(x) \neq y].
\]
This metric represents the proportion of misclassified points in \( \mathcal{L} \). For completeness, we set \( \text{er}_{\emptyset}(h) = 0 \). When \( \mathcal{L} = \mathcal{Z}_m \), where \( \mathcal{Z}_m \) represents the first \( m \) labeled data points,we abbreviate:

```
 \text{er}_m(h) = \text{er}_{\mathbb{Z}_m}(h).    
```



We also define the *version space*, 
\[
V_m^* = \{ h \in \mathcal{C} : \forall i \leq m, h(X_i) = f^*(X_i) \},
\]
which includes all classifiers consistent with the first \( m \) data points \( \{X_1, \ldots, X_m\} \).

For a given set of classifiers \(\mathcal{H}\) and any \(\varepsilon \in [0, 1]\), define the \(\varepsilon\)\(-minimal\) set as:
\[
\mathcal{H}(\varepsilon) = \{h \in \mathcal{H}: er(h)\leq \varepsilon\}.
\]

Additionally, for any classifier \(h\), the \(\varepsilon\)\(-ball\) centered at \(h\) is defined as:
\[
\mathcal{B}_{\mathcal{H}, \textnormal{P}_{X}}(h, \varepsilon) = \{g \in \mathcal{H}: \textnormal{P}_{X}(x: g(x) \neq h(x)) \leq \varepsilon\}.
\]

When \(\mathcal{H}\) is the hypothesis class \(\mathcal{C}\), we abbreviate \(\mathcal{B}_{\textnormal{P}_{X}}(h, \varepsilon)\) as \(\mathcal{B}_{\mathcal{C}, \textnormal{P}_{X}}(h, \varepsilon)\), and if \(\textnormal{P}_{X}\) is implicit, we write \(\mathcal{B}_{\mathcal{H}}(h, \varepsilon)\) for \(\mathcal{B}_{\mathcal{H}, \textnormal{P}_{X}}(h, \varepsilon)\). Moreover, the radius of the set \(\mathcal{H}\), denoted \(\text{radius}(\mathcal{H})\), is the smallest \(\varepsilon\) such that \(\mathcal{H} = \mathcal{B}_{\mathcal{H}}(f^*, \varepsilon)\), where \(f^*\) is the optimal classifier (i.e. the perfect classifier). Lastly, define the region of disagreement of \(\mathcal{H}\) as:

\[
\text{DIS}(\mathcal{H}) = \{x \in \mathcal{X}: \exists h, g \in \mathcal{H} \text{ such that } h(x) \neq g(x)\}, 
\]
the set of points where different classifiers within the set 
`\mathcal{H}` disagree about the correct label, as in [hanneke2014theory].

The particular algorithm for the realizable cases that will be investigated in this thesis belongs to a specific group called "*disagreement-based"* active learning. As [hanneke2014theory] puts it, they have a mechanism that step by step update a set, `V`, which contains candidate classifiers—one of which will ultimately be chosen. Unlabeled samples are processed sequentially, and the algorithm requests labels for those samples, `Y_i`, only when there is a lack of consensus in `V` about the predicted label for sample `X_i`. The set `V` is updated periodically by eliminating classifiers that perform poorly based on the labels obtained [hanneke2014theory]. 

In other words, as the author has explained in [hanneke2014theory], the label complexity of this approach involves analyzing the characteristics of the disagreement regions, `\text{DIS}(V)`, that are formed during execution. Specifically, since labels are only requested for samples in `\text{DIS}(V)`, it is essential to assess the likelihood, `\textnormal{P}_{X}(\text{DIS}(V))`, that a randomly chosen sample falls into this disagreement region. As we will demonstrate, it is often practical to set a simple bound on the radius of `V` for the sets encountered in these algorithms. Consequently, to derive a straightforward bound on label complexity, it can be useful to approximate `\textnormal{P}_{X}(\text{DIS}(V))` by a linear function of the radius of `V`. The disagreement coefficient is formally defined as follows:

\begin{definition} [{[hanneke2014theory]}]
For any \( r_0 \geq 0 \) and a classifier \( h \), the disagreement coefficient of \( h \) with respect to a hypothesis class \( \mathcal{C} \) and distribution \( \textnormal{P}_{X} \) is given by: 
\[
\theta_h(r_0) = \sup_{r > r_0} \frac{\textnormal{P}_{X}(\textnormal{DIS}(\textnormal{B}(h, r)))}{r} \vee 1,
\]
where \( \textnormal{DIS}(\textnormal{B}(h, r)) \) represents the region where classifiers in \( \mathcal{C} \) disagree on labels within radius \( r \) of \( h \). When \( h = f^* \), this is simplified as \( \theta(r_0) = \theta_{f^*}(r_0) \), referred to as the disagreement coefficient for the class \( \mathcal{C} \) under the distribution \( \textnormal{P}_{X} \).
    
\end{definition}

The disagreement coefficient quantifies the rate at which the region of disagreement among classifiers diminishes as their uncertainty decreases. A low disagreement coefficient signifies that the classifiers' predictions converge rapidly with reduced error, suggesting a lower requirement for label requests. Conversely, a high disagreement coefficient indicates a slower convergence, necessitating more label queries.

The coefficient is equal to zero when there is no disagreement among classifiers, even with the largest possible radius of the \(\varepsilon\)\(-ball\). This implies that the classifiers are perfectly consistent in their predictions. Conversely, it becomes one when there is always disagreement among classifiers, regardless of the radius of the \(\varepsilon\)\(-ball\). This indicates that the classifiers are highly diverse and have fundamentally different ways of making predictions.


We also write small-`o` notation to express asymptotic dependences. So, the statements like:
\[
\lim_{\varepsilon \rightarrow 0}\frac{u(\varepsilon)}{v(\varepsilon)} = 0,
\]

could be written as:
```
    u(\varepsilon) = o(v(\varepsilon)). 
```

The equation \ref{o(v)} means that `u(\varepsilon)` should be smaller than `v(\varepsilon)`. See Figure\ref{fig:small_o} please.






We will see the application of a function \(o(v(\varepsilon))\) later during determining `n`, the labeling budget.

We use the notation `u(\varepsilon, \delta) \lesssim v(\varepsilon, \delta)`, means that there exists a universal constant `c \in (0, \infty)` such that `u(\varepsilon, \delta) \leq c v(\varepsilon, \delta)` for all `\varepsilon, \delta \in (0, 1)`. Similarly, `u(\varepsilon, \delta) \gtrsim v(\varepsilon, \delta)` indicates that `u(\varepsilon, \delta) \geq c v(\varepsilon, \delta)` for all `\varepsilon, \delta \in (0, 1)`, where `c \in (0, \infty)` is an implicit universal constant. In this context (later we will state Theorem \ref{Theorem: o(v)} regarding the properties that lead to choosing `o(v(\varepsilon))`), we define \( \lambda^k \) as the Lebesgue measure on \( \mathbb{R}^k \). For a probability measure \( \textnormal{P}_{X} \) on \( \mathbb{R}^k \) to have a density \( p : \mathbb{R}^k \to [0, \infty] \) (relative to \( \lambda^k \)), \( p \) must be a measurable function. For any measurable subset \( \textnormal{A} \subseteq \mathbb{R}^k \), the probability \( \textnormal{P}_{X}(\textnormal{A}) \) is then given by
\[
\textnormal{P}_{X}(\textnormal{A}) = \int \mathbf{1}_{\textnormal{A}}(x) p(x) \, \lambda^k(dx).
\]

According to the Radon-Nikodym theorem, \( \textnormal{P} \) possesses a density if and only if every measurable subset \( \textnormal{A} \subseteq \mathbb{R}^k \) with \( \lambda^k(A) = 0 \) also satisfies \( \textnormal{P}_{X}(\textnormal{A}) = 0 \).


The following notations also apply. The `x` represents a specific data point or an individual instance from the instance space `X`. It is a generic variable used to denote any arbitrary data point. `X_i`, however, refers to a specific instance within a sequence of unlabeled data points. Each `X_i` is one element in a sequence `\{X_1, X_2,...\}`, where `i` typically indexes individual samples in this sequence. Additionally, we use the notation `\text{Log}(x) = \max\{\ln(x), 1\}` for `x \geq 0`.


In this work for realizable cases, we will use linear classifiers, namely:


``` 
\mathcal{C}=\{h: \mathbf{w} \in [0, 1]^k, \mathbf{b} \in [0, 1]\}
```
where,
``` 
h(x) =    \mathbf{w}^{T} \cdot x + \mathbf{b}.
```

Furthermore, by defining the sign function as follows:

``` 
\textnormal{sign}(a) = \left\{ \begin{array}{lcl}
1 & \text{if } a > 0\\ 0 & otherwise,\\
                \end{array}\right. \notag
```
and plugging in \ref{linear classifier} in sign function we get:

``` 
\textnormal{sign}(h(x)) = \left\{ \begin{array}{lcl}
1 & \text{if } \mathbf{w}^{T} \cdot x + \mathbf{b}> 0\\ 0 & otherwise.\\
                \end{array}\right. \notag
```

Now, we are ready to dig into the algorithm and supporting theorems in detail.


## Algorithm and Theorems

One of the well-known works that has studied the realizable case is the paper written by [cohn1994improving]. This work is famous for its algorithm so-called CAL named after the authors. There are several equivalent ways that this algorithm has been described. Here, as in [hanneke2014theory], to simplify and keep it more coherent the version of pseudo-code that is based on version space will be used. In the following, we will explain how each step of the procedure works in this algorithm. After that, a precise mathematical analysis for characterizing it will be provided.

\begin{algorithm}
\caption{CAL(`n`)}
\begin{algorithmic}[1]
\State `m \leftarrow 0`, `t \leftarrow 0`, `V \leftarrow \mathcal{C}`
\While{`t < n` and `m < 2^n`}
    \State `m \leftarrow m + 1`
    \If{`X_m \in \text{DIS}(V)`}
        \State Request label `Y_m`; let `V \leftarrow \{ h \in V : h(X_m) = Y_m \}`, `t \leftarrow t + 1`
    \EndIf
\EndWhile
\State \Return any `\hat{h} \in V`
\end{algorithmic}
\end{algorithm}


In the first line three parameters \(m, t, and \ V\), corresponding to respectively, the number of iterations, the number of queried labels, and the version space is initialized. Then it begins the loop and proceeds next steps as long as both of \(t\ and\ m\) simultaneously meet the criteria that are being less than query budget, \(n\), for \(t\), and similarly being less than the iteration threshold, \(2^n\), for \(m\). We can see that the iteration threshold is the function of the query budget.
Afterward, inside the *while* loop, \(m\) will be incremented by one. Then in the *line 4*, the condition of the disagreement region is checked. That is, whether the sample \(X_m\) is a member of this set or not. If this constraint is not satisfied by \(X_m\), it means that so far there were not any classifiers that label \(X_m\) differently. Namely, 
\[\nexists \ h,h'\in H\  s.t. \ h(X_m) \neq h'(X_m).\]

On the other hand, if the sample \(X_m\) violates this condition, the cascade of the subsequent updates will occur:
\begin{itemize}
    \item algorithm will request label \(Y_m\) from the oracle.
    \item \(V\), the version space n q will be updated such that:
    \[\forall \ h,h'\in H:\ h(X_m) = h'(X_m) = Y_m.\]
    \item \(t\), the counter of query budget will be incremented to \(t+1\)
\end{itemize} 

The moment that the logical condition of *while* loop has been violated by either exceeding the query budget or iteration threshold, the algorithm will go out of the loop and implement the last step that is returning the \(\hat{h} \in V.\)




In the following the related theorems for characterizing the mechanism of algorithm \ref{alg_cal} will be expressed.


\begin{theorem}[[hanneke2014theory]]

CAL achieves a label complexity \(\Lambda\) such that, for \(\textnormal{P}_{X,Y}\) in the realizable case, \( \forall \varepsilon, \delta \in (0,1)\),
\[
\Lambda(\varepsilon, \delta, \textnormal{P}_{X,Y}) \lesssim 
\theta(\varepsilon) \bigg( d\textnormal{Log}(\theta(\varepsilon)) + \textnormal{Log} \bigg(\frac{\textnormal{Log}(1/\varepsilon)}{\delta} \bigg) \bigg)\textnormal{Log}(1/\varepsilon).
\]
\end{theorem}

This theorem provides an upper bound on the label complexity of the CAL algorithm. Here, the label complexity `\Lambda` measures the number of labeled examples needed by the active learning algorithm to achieve a certain level of accuracy within `\varepsilon` with high probability (at least `1-\delta`).

If we note to the Algorithm \ref{alg_cal}, we can see that to run CAL we need `n`, the labeling budget, and the initial hypotheses class `\mathcal{C}`. So before running CAL, utilizing Theorem \ref{Theorem:label complexity in realizable case} we can find the minimum labeling budget. The VC dimension `d` and the disagreement coefficient `\theta(\varepsilon)` are also two key components to fulfill mapping `\varepsilon, \delta`, and `\textnormal{P}_{X,Y}` to the labeling budget `n`. As mentioned in the previous section, we are going to use a class of linear hypotheses. In the context of linear classifiers, the determination of the Vapnik-Chervonenkis (VC) dimension (Theorem \ref{Theorem: VC dimension}) and the disagreement coefficient (Theorem \ref{Theorem: o(v)}) is grounded in established theoretical frameworks and we will state those without proof. Later during implementation, we will use these theorems in practice.
 

\begin{proof}[**Proof of Theorem \ref{Theorem:label complexity in realizable case**}][hanneke2014theory] The proof includes five steps as following:


\noindent**Step 1: Setting Parameters**

First we set parameters by selecting `\varepsilon` and `\delta` such that both lay within the interval `(0, 1)`, then execute the CAL algorithm with a label budget `n \in \mathbb{N}`, chosen to be sufficiently large:
\[
n \geq \log_2(2/\delta) + 8ec' \theta(\varepsilon) \left( d \log(\theta(\varepsilon)) + 2 \log\left(\frac{2 \log_2(4/\varepsilon)}{\delta}\right) \right) \log_2(2/\varepsilon).
\]

This ensures that the number of label requests is large enough to meet the desired error and confidence levels.


\noindent**Step 2: Version Space Definition**

Now we define key concepts.
Set `M \subseteq \{0, \ldots, 2^n\}` describe the range of values
`m` can get while CAL runs. For each `m \in M`, let `V_m` 
as the version space at that point, containing all hypotheses consistent with the labels seen so far. Assume an even `E`, of probability 1, such that for every `m \in \mathbb{N}` has `Y_m = f^*(X_m)`. By induction, the version space `V_m`  at each step `m` contains all classifiers that agree with the observed labels, i.e., `V_m = V^{*}_{m}`. Formally:
\[
\forall m \in M, f^* \in V_m = V_m^* = \{h \in \mathcal{C} : er_m(h) = 0\}.
\]

Please note that to create the above `V^{*}_{m}`, which is equivalent to check the `er_m(h) = 0 ` criteria, mean to check the `h(x) = y`. Hence, in reality, we are unaware of `y`. Thus, the practical alternative would be exploiting the fact that the optimal hypothesis, `f^{*}`, is always in version space and comparing the prediction of other hypotheses with it.


\noindent**Step 3: Setting the Fundamental Error Constraints**

To proceed, we incorporate Lemma \ref{lemma: Lemma_1} to establish the fundamental bounds necessary for our proof:

\begin{lemma}[[hanneke2014theory]] 
There exists a universal constant `c \in (1, \infty)` such that, for any `\gamma \in (0, 1),\ a \in [1, \infty),\ \alpha \in [0, 1],` and any `m \in \mathbb{N}`, with `\varepsilon_m = \left(\frac{ad}{m}\right)^{\frac{1}{2-\alpha}}`, and

```
U(m, \gamma) = c \ \min\left\{ \begin{array}{lcl}
\left( \frac{a \left( d \textnormal{Log} (\theta(a \varepsilon_m^{\alpha})) + \textnormal{Log}(1/\gamma) \right)}{m} \right)^{\frac{1}{2 - \alpha}}\\ \frac{d \textnormal{Log} (\theta(d/m)) + \textnormal{Log}(1/\gamma)}{m} + \sqrt{\frac{\nu \left( d \textnormal{Log}(\theta(\nu)) + \textnormal{Log}(1/\gamma) \right)}{m}}\\
                \end{array}\right. \notag    
```
with probability at least \( 1 - \gamma \), \( \forall h \in \mathcal{C} \), the following inequalities hold:
\[
er(h) - er(f^*) \leq \max \left\{ 2 (er_m(h) - er_m(f^*)), U(m, \gamma) \right\},
\]
\[
er_m(h) - \min_{g \in \mathcal{C}} er_m(g) \leq \max \left\{ 2 (er(h) - er(f^*)), U(m, \gamma) \right\},
\]
where `er_{m}(h)` is defined as in \ref{er_m}, and `d= \textnormal{vc}(\mathcal{C})`. 
\end{lemma}


This lemma provides a bound, `U(m, \lambda)` on the maximum discrepancy between empirical error `er_{m}(h)` and true error `er(h)`, allowing us to understand how close our model is to the true error rate after observing a certain number of labeled samples. It ensures that any hypothesis selected from the hypothesis class maintains a small error compared to an optimal or true hypothesis with a high probability `1-\lambda`. Given a desired error threshold `\varepsilon`, Lemma \ref{lemma: Lemma_1} helps us solve for the smallest `m` such that `U(m, \lambda) \leq \varepsilon`, ensuring the error bound is met with high probability. So, by Lemma \ref{lemma: Lemma_1} we can establish a mechanism between the minimum number of samples in the worst-case scenario to guarantee with a high probability, here `1-\lambda`, that the error is under control during executing the Algorithm \ref{alg_cal}.

It is worth mentioning that the noise control parameters, namely, `\alpha,\ \nu,` and `a` should be chosen such that the setting meets the realizable case environment. That is to say, these parameters would be `\alpha= 1,\ a= 1,` and `\nu= 0`. By having some computation we can find the minimum `m` such that the `U(m, \lambda)` is guaranteed to be less than a specific error threshold. Namely, for some global fixed ` c' \in [1, \infty ) `, for any `m \in \mathbb{N}`, and `\varepsilon, \lambda \in (0, 1)`, we have:
```
    m \geq c' \left( \frac{\nu+ \varepsilon}{\varepsilon^2} \right) \left( d \textnormal{Log}(\theta(\nu + \varepsilon)+ \textnormal{Log}(1/\lambda)\right) \Longrightarrow U(m, \lambda) \leq \varepsilon,  
```
and by plugging in `\alpha= 1, a= 1,` and `\nu= 0`, \ref{lemma_1 ini result_2} becomes:
```
    m \geq c' \left( \frac{1}{\varepsilon} \right) \left( d \textnormal{Log}(\theta(\varepsilon)+ \textnormal{Log}(1/\lambda)\right) \Longrightarrow U(m, \lambda) \leq \varepsilon.  
```


\noindent**Step 4: Error Bound Across Iterations**

Now define `i_\varepsilon = \lceil \log_2(1/\varepsilon) \rceil`, as the number of iterations to achieve the error `\varepsilon` and set `I = \{0, \ldots, i_\varepsilon\}`. At the end of each iteration, the version space is updated. For `i \in I`, let `\varepsilon_i = 2^{-i}`; this represents the decreasing error thresholds for each step. furthermore, let `m_0 = 0`, and for `c'` as in \ref{lemma_1 result_2}, for each `i \in I \setminus \{0\}`, define
\[
m_i = \left\lceil c' \left( \frac{1}{\varepsilon_i} \right) \left( d \textnormal{Log}(\theta(\varepsilon_i)) + \textnormal{Log}\left( \frac{2(2 + i_\varepsilon - i)^2}{\delta} \right) \right) \right\rceil.
\]

\noindent**Step 5: Bounding Error Probability**

By Lemma \ref{lemma: Lemma_1}, and \ref{lemma_1 result_2}, and involving a union bound we show that, with high probability, the maximum error of any classifier in the version space is below a certain threshold for all iterations. Let's break it down in detail. The objective is to show that, with high probability, for every iteration `i` in the set `I`, the supremum (maximum) error rate of hypotheses in the version space
`V^{*}_{m}` is at most `\varepsilon_i`. In other words, regarding the error probability for individual iteration, namely, `i \in I`, we want to bound the probability that the error rate `\sup_{h \in V_{m_i}^*} er(h) \leq \varepsilon_i` exceeds `\varepsilon_i`.
From Lemma \ref{lemma: Lemma_1} we have:
\[
	\textnormal{P}_{X,Y} \bigg(\sup_{h \in V^{*}_{m_i}} er(h) > \varepsilon_i \bigg) \leq \frac{\delta}{2(2+i_\varepsilon-i)^2}.
\]

This inequality states that the probability of having an error rate greater than `\varepsilon_i` for any hypothesis in `V_{m_i}^*` is bounded by a decreasing function of `i`. Then by the union bound, we combine the probabilities of multiple events. It states that the probability of the union of several events is at most the sum of the probabilities of those events. Applying the union bound here allows us to combine the error probabilities for all iterations `i \in I/\{0\}`. We sum the probabilities from Lemma \ref{lemma: Lemma_1} across all iterations:
```
  	\textnormal{P}_{X,Y} \bigg(\bigcup_{i=1}^{i_{\varepsilon}}\sup_{h \in V_{m_i}^*} er(h) \geq \varepsilon_i \bigg) \leq  \sum_{i=1}^{i_{\varepsilon}}\frac{\delta}{2(2+i_\varepsilon-i)^2} .
```

This sum captures the total probability that any of the iterations `i` will have an error rate exceeding `\varepsilon_i`.
Based on  the fact that the series `\sum_{i=1}^{i_{\varepsilon}}\frac{1}{(2+i_\varepsilon-i)^2}` converges to a value less than 1, we can bound the left-hand side of the inequality \ref{sum of errors}:
```
  \sum_{i=1}^{i_{\varepsilon}}\frac{\delta}{2(2+i_\varepsilon-i)^2}  \leq \frac{\delta}{2}. 
```
This ensures that the combined probability across all iterations is still small and manageable. By combining (\ref{sum of errors})
and (\ref{bounding sum of erros}) we can see:

```
    	\textnormal{P}_{X,Y} \bigg(\bigcup_{i=1}^{i_{\varepsilon}}\sup_{h \in V_{m_i}^*} er(h) \geq \varepsilon_i \bigg) \leq  \frac{\delta}{2}.  
```

Now we can define an event over the intersection of classifiers in `V_{m_i}^*`. Let's call this even `E_{\delta}`. Please note that `E_{\delta}` is the completeness of the inequality in (\ref{sum of errors}). Namely, it can be written as follows:
```
  	\textnormal{P}_{X,Y} \bigg(\bigcap_{i=1}^{i_{\varepsilon}}\sup_{h \in V_{m_i}^*} er(h) \leq \varepsilon_i \bigg) &=  1- 	\textnormal{P}_{X,Y}\bigg(\bigcup_{i=1}^{i_{\varepsilon}}\sup_{h \in V_{m_i}^*} er(h) \geq \varepsilon_i \bigg) \notag \\
  &= 1 - \sum_{i=1}^{i_{\varepsilon}}{	\textnormal{P}_{X,Y} \bigg(\sup_{h \in V_{m_i}^*} er(h) \geq \varepsilon_i \bigg)} \notag\\
  & \geq 1- \sum_{i=1}^{i_{\varepsilon}}\frac{\delta}{2(2+i_\varepsilon-i)^2} \notag \\ 
  & \geq 1 - \frac{\delta}{2}, 
```
this guarantees that with probability at least `1-\frac{\delta}{2}`
the error rates stay within the desired bounds for each step `i`.

It is worth mentioning that the CAL will process every new point to check whether it falls into the most recent corresponding version space's disagreement area. For event `E`, the total number of these requested labels up to the `m_{i_{\varepsilon}}` visited point is:
```
     \sum_{m=1}^{\min\{m_{i_\varepsilon}, \max M\}} \mathbf{1}_{\textnormal{DIS}(V_{m-1})}(X_m) = \sum_{m=1}^{\min\{m_{i_\varepsilon}, \max M\}} \mathbf{1}_{\textnormal{DIS}(V_{m-1}^*)}(X_m), 
```

The summation on the left-hand side  of the (\ref{sum of requested labels}) also can be brake down to iteration, namely:
```
   \sum_{m=1}^{m_{i_\varepsilon}}\mathbf{1}_{\textnormal{DIS}(V_{m-1}^*)}(X_m) = \sum_{i \in I/\{0\}} \sum_{m= m_{i-1}+1}^{m_i} \mathbf{1}_{\textnormal{DIS}(V_{m-1}^*)}(X_m), 
```
where, `\mathbf{1}_{\textnormal{DIS}(V_{m-1})}(X_m)`, means if `X_m` is in the disagreement set.




The iteration `i`, corresponds to a particular threshold `\varepsilon_i`, that decreases exponentially. During each iteration `i`, the CAL algorithm requests labels for instances within a specific segment of indices, from `m_{i-1}+1` to `m_i`. This range can be express like `m \in \{m_{i-1}+1,...,m_i \}`. These are actually processed points between two consecutive labels requested by CAL. The endpoints `m_{i-1}` and `m_i` are chosen such that the number of label requests in each iteration is sufficient to meet the error threshold `\varepsilon_i`. In addition, the version space `V_{m}^*` is monotonically decreasing, meaning it gets smaller as more labels are requested, that is  `V_{m_i}^* \subseteq V_{m_{i-1}}^*`. This monotonicity plays a crucial role in bounding the number of label requests, as it implies that once a point leaves the disagreement set, it never re-enters it. Please also observe that the disagreement area of the specific version space for points processed during iteration `i` stays the same, for instance, `m \in \{m_{i-1}+1,...,m_i \}` have the same related disagreement area of the version space which particularly it is `V_{m_{i}}^*`.  From the monotonicity of `V_{m}^*`, every `m` and `i` have `\textnormal{DIS}(V_{m-1}^*) \subseteq \textnormal{DIS}(V^{*}_{m_{i-1}})`. Thus, (\ref{union of errors}) conveys that on event `E_\delta`, `V^{*}_{m_{i-1}} \subseteq \textnormal{B}(f^*, \varepsilon_{i-1})`, and accordingly `\textnormal{DIS}(V^{*}_{m_{i-1}}) \subseteq \textnormal{DIS}(\textnormal{B}(f^*, \varepsilon_{i-1}))`.


Hence, (\ref{sum of requested labels 2}) is at most
```
    \sum_{i \in I \setminus \{0\}} \sum_{m=m_{i-1}+1}^{m_i} \mathbf{1}_{\textnormal{DIS}(\textnormal{B}(f^*, \varepsilon_{i-1}))}(X_m). 
```

To bound the  `\sum_{m=1}^{m_{i_\varepsilon}}\mathbf{1}_{\textnormal{DIS}(V_{m-1}^*)}(X_m)`, we use the fact that the indicators are Bernoulli random variables representing whether a point `X_m` is in the disagreement area of version space at step `m-1`. The expected value of each indicator `\mathbf{1}_{\textnormal{DIS}(V_{m-1}^*)}(X_m)`, equals the probability that `X_m` lies within the disagreement set. This probability can be upper-bounded using the size of the version space and the properties of the disagreement set.
Thus, the sum of `m_{i_\epsilon}` independent Bernoulli random variables, have the following expected value:
\[
\sum_{i \in I \setminus \{0\}} (m_i - m_{i-1}) \textnormal{P}_{X}(\textnormal{DIS}(\textnormal{B}(f^*, \varepsilon_{i-1}))). 
\]


Consequently, by applying the *Chernoff bound*, we show that with high probability, on an event `E'_\delta` of probability at least `1 - \delta/2`, the term in (\ref{sumsumX_m}) is at most:
```
    \log_2(2/\delta) + 2e \sum_{i \in I \setminus \{0\}} (m_i - m_{i-1}) \textnormal{P}_{X}(\textnormal{DIS}(\textnormal{B}(f^*, \varepsilon_{i-1}))). 
```

\noindent**Step 6: Concluding the Proof**

According to the definition of `m_i`, as `i` increases, `\varepsilon_i = 2^{-i}` decreases exponentially. The expression inside the ceiling function is positive and increases as `i` increases because the term ` 1/\varepsilon_i` increases exponentially as `i` increases. Although `\textnormal{Log}(\theta(\varepsilon_i))` and `\textnormal{Log}(\frac{2(2+ i_{\varepsilon} -i)^2}{\theta}))` are growing more slowly and logarithmically, they are positive and contribute to the overall increase. Since `c'` is a positive constant the overall expression inside the ceiling function increases.
By definition of the disagreement coefficient we know that, `\textnormal{P}_{X}(\textnormal{DIS}(\textnormal{B}(f^*, \varepsilon_{i-1})))\leq \theta(\varepsilon_{i-1}) \varepsilon_{i-1}`. Then by integrating this with the fact that `\forall i, m_{i-1} \leq m_i`, for `i \in I \setminus \{0\}`, `m_i \textnormal{P}_{X}(\textnormal{DIS}(\textnormal{B}(f^*, \varepsilon_{i-1})))` is at most
```
    4c' \theta(\varepsilon_{i-1}) \left( d \textnormal{Log}(\theta(\varepsilon_i)) + \textnormal{Log}\left( \frac{2(2 + i_\varepsilon - i)^2}{\delta} \right) \right). 
```

Consequently, in the meantime `\theta(\varepsilon_{i-1}) \leq \theta(\varepsilon_i) \leq \theta(\varepsilon)` for `i \in I \setminus \{0\}`, \ref{Chernoff result} is less than
\[
\log_2(2/\delta) + 8ec' \theta(\varepsilon) \left( d \textnormal{Log}(\theta(\varepsilon)) + 2 \textnormal{Log}\left( \frac{2 \log_2(4/\varepsilon)}{\delta} \right) \right) \log_2(2/\varepsilon) \leq n.
\]

Specifically, we have demonstrated that during the event `E \cap E_\delta \cap E'_\delta`, \(\sup_{h \in V^{*}_{m_{i_\varepsilon}}} er(h) \leq \varepsilon_{i_\varepsilon} \leq \varepsilon\), and the cumulative quantity of label requests initiated by the CAL while `m \leq m_{i_\varepsilon}` is less than `n; \text{ since } 2^n > m_{i_\varepsilon}`, This crucial factor implies that we absolutely require `\max M \geq m_{i_\varepsilon}`, so that  `\hat{h} \in V^{*}_{m_{i_\varepsilon}}`, and thus  `er(\hat{h}) \leq \varepsilon`.
Taking into account that `E \cap E_\delta \cap E'_\delta` has probability at least `1 - \delta` (by means of union bound), and that
```
    \log_2(2/\delta) + 8ec' \theta(\varepsilon) \left( d \textnormal{Log}(\theta(\varepsilon)) + 2 \textnormal{Log}\left( \frac{2 \log_2(4/\varepsilon)}{\delta} \right) \right) \log_2(2/\varepsilon) \notag\\
\lesssim \theta(\varepsilon) \left( d \textnormal{Log}(\theta(\varepsilon)) + \textnormal{Log}(\textnormal{Log}(1/\varepsilon)/\delta) \right) \textnormal{Log}(1/\varepsilon), \notag
```
demonstrates the validity of the proof.\end{proof}



In the realizable case, there are two other important measures worth inspecting. The first one is the number of labels CAL would request among the first \(m\) unlabeled data points, namely, it is 
\[
for \ n,m \in \mathbbm{N}, \ 
N(m) = \sum_{i=1}^{m} \mathbf{1}_{\textnormal{DIS}(V^{*}_{i-1})}(X_i),
\]
and, the other one is the number of unlabeled data points CAL would investigate up to its \(n\)-th label request, namely:
\[
M(n) = min\{ k \in \mathbb{N}: N(k) = n\} \cup \{\infty \}.
\]
The Theorems (\ref{number of label request}, \ref{Theorem: vc probability reduction}) will formally characterize these properties. For extended proofs please refer to the [hanneke2012activized].

\begin{theorem}[[hanneke2014theory]]

For any \(m \in \mathbb{N} \cup \{0\}\) and \(r \in (0, 1 )\),
\[
\mathbb{E}[\textnormal{P}_{X}(\textnormal{DIS}(V_{m}^{*}))] \geq (1-r)^m \textnormal{P}_{X}(\textnormal{DIS}(\textnormal{B}(f^{*}, r))).
\]
 This also implies that for any \( \varepsilon \in (0, 1) \),
 \[
 \mathbb{E}[N(\lceil 1/\varepsilon \rceil)] \geq \theta(\varepsilon)/2.
 \]
\end{theorem}
According to this theorem, we can assess the relative performance of the algorithm CAL using specific criteria. Requesting more labels when `m` is low is generally advantageous. Early in the process, the version space `V^{*}_m` is large because the algorithm has observed fewer labeled samples, encompassing a wide range of classifiers consistent with the limited data. Consequently, the disagreement region is extensive since the algorithm has not yet refined the classifier subset.

Frequent label requests at low `m` indicate that the algorithm is encountering numerous samples within this disagreement region. This is beneficial as it reduces uncertainty early, rapidly shrinking the version space by eliminating classifiers inconsistent with the observed labels in high-uncertainty areas. This accelerates convergence toward an optimal classifier, enabling the algorithm to make more informed decisions as it progresses.

\begin{proof}[**Proof of Theorem \ref{number of label request**}][hanneke2012activized] The proof consists of following steps:


\noindent**Step 1: Setting Definition**

Let's put \( D_m = \textnormal{DIS}(V^{*}_{m} \cap \textnormal{B}(f^*, r))\); This is a set of points that are in the disagreement area where classifiers belong to both *version space* and \(r-ball\) centered at \(f^*\). On the other hand, we know that \(V^{*}_{m} \subseteq \textnormal{B}(f^*, r) \). Thus, 
\( D_m = \textnormal{DIS}(V^{*}_{m} \cap \textnormal{B}(f^*, r)) = \textnormal{DIS}(V^{*}_{m})\). Please note that none of the \(m\) previously observed points belong to the \(D_m\), namely, for all \(m\) previously processed points
 \( \textnormal{P}_{X}(\textnormal{DIS}(V^{*}_{m})) = 0\). Therefore, the probability of occurrence of \(\textnormal{P}_{X}(\textnormal{DIS}(V^{*}_{m}))\) means to investigate the area where the rest of the unobserved points will happen concerning \(V^{*}_{m}\) and \( \textnormal{B}(f^*, r) \).


\noindent**Step 2: Formulating Expected Value**

Now let us put:
```
\mathbb{E} \left[ 
\sum_{m=1}^{\lceil 1/r \rceil} \mathbf{1}_{D_{m-1}}(X_m) 
\right] &= \sum_{m=1}^{\lceil 1/r \rceil} \mathbb{E} \left[X_m \in D_{m-1} | V^{*}_{m-1}\right]\notag \\
&= \sum_{m=1}^{\lceil 1/r \rceil} \mathbb{E} [\textnormal{P}_{X}(D_{m-1})],
```
and make lower bound for \(\mathbb{E} [\textnormal{P}_{X}(D_{m-1})]\) for \(m \in \mathbb{N} \cup \{0\}\).


\noindent**Step 3: Lower Bound for Expected Probability**

Please observe that in general if any \( x \in \textnormal{B}(f^*, r)\), by definition there could be one or several \(h_x \in \textnormal{B}(f^*, r) \) such that the \(\textnormal{P}_{X}( x: h_x(x) \neq f^{*}(x)) \leq r\) and in case that the \(h_x \in V^{*}_{m}\), this leads to the fact that \(x \in D_m.\) So, we have:
 \[
 \forall x, \mathbf{1}_{D_m}(x) \geq \mathbf{1}_{\textnormal{DIS}(\textnormal{B}(f^*, r))}(x). \mathbf{1}_{V^{*}_{m}}(h_x)
 .\]
 
 To convert the \(\mathbf{1}_{V^{*}_{m}}(h_x)\) as a function of \(x\), we need to interpret the meaning of it in terms of the disagreement area. Hence, if \(h_x \in V^{*}_{m} \) by definition it means that for all \( i \leq m\), \(X_i\)'s should stay in area where \(h_x(X_i) = f^{*}(X_i)\). This implies that every \(i \leq m\), \(X_i\)'s should belong to **complement** area of where \(h_x(X_i) \neq f^{*}(X_i)\), or equivalently, \( \prod_{i=1}^{m}\mathbf{1}_{\textnormal{DIS}(\{h_x, f^*\})^{c}}(X_i)\). Accordingly, we have:
 \[
 \forall x, \mathbf{1}_{D_m}(x) \geq \mathbf{1}_{\textnormal{DIS}(\textnormal{B}(f^*, r))}(x). \mathbf{1}_{V^{*}_{m}}(h_x) = \mathbf{1}_{\textnormal{DIS}(\textnormal{B}(f^*, r))}(x) . \prod_{i=1}^{m}\mathbf{1}_{\textnormal{DIS}(\{h_x, f^*\})^{c}}(X_i)
 .\]
 
 Now, again according to [hanneke2012activized]:

```
% &{\mathbb{E}(P(\mathbf{DIS}(V^{*}_{m})))}\\
\mathbb{E}(\textnormal{P}_{X}({D_{m}}))
&= \textnormal{P}_{X}(X_{m+1} \in ({D_{m}}))
= \mathbb{E} \left[ \mathbb{E} \left[ \mathbf{1}_{{D_m}}(X_{m+1}) \bigg| X_{m+1}\right] \right]\notag \\
& \geq \mathbb{E} \left[ \mathbb{E} \left[\mathbf{1}_{\textnormal{DIS}(\textnormal{B}(f^*, r))}(X_{m+1}). \prod_{i=1}^{m}\mathbf{1}_{\textnormal{DIS}(\{h_x, f^*\})^{c}}(X_i) \bigg| X_{m+1}\right] \right] \notag\\
&= \mathbb{E} \left[ \prod_{i=1}^{m} \textnormal{P}_{X}\biggl(h_{X_{m+1}}(X_i) = f^{*}(X_i) \bigg| X_{m+1} \biggl)\mathbf{1}_{\textnormal{DIS}(\textnormal{B}(f^*, r))}(X_{m+1})\right] \notag\\ %_{X}(DIS(B(f^*,r)))}\\
& \geq \mathbb{E} \biggl[ (1 - r)^{m}. \mathbf{1}_{\textnormal{DIS}(\textnormal{B}(f^*, r))}(X_{m+1}) \biggr] \notag \\
&= \biggl(1 - r\biggr)^{m}\textnormal{P}_{X}(\textnormal{DIS}(\textnormal{B}(f^*, r))).
```


\noindent**Step 4: Conclusion**

We know the following about geometric series :
\[
\sum_{m=1}^{\lceil 1/r \rceil}(1 - r)^{m-1} = \frac{1 - (1-r)^{\lceil 1/r \rceil} }{r}, 
\]
and also we have \( \bigl( 1 -r \bigr)^{\lceil 1/r\rceil} \geq \frac{1}{e}\) for \(r \in (0, \epsilon)\) for small \(\varepsilon > 0\). Finally, by plugging in the last term in \ref{(1-m)^m.P(DIS(B(f^*,r)))} into the \ref{exp(p(D_m_1))} we will have:

```
\sum_{m=1}^{\lceil 1/r \rceil}(1 - r)^{m-1}\textnormal{P}_{X}(\textnormal{DIS}(\textnormal{B}(f^*, r))) &= \Bigl( 1 - (1-r)^{\lceil 1/r \rceil} \Bigr)\frac{\textnormal{P}_{X}(\textnormal{DIS}(\textnormal{B}(f^*, r)))}{r} \notag \\
& \geq \left( 1 - \frac{1}{e} \right) \frac{\textnormal{P}_{X}(\textnormal{DIS}(\textnormal{B}(f^*, r)))}{r} \notag \\
& \geq \frac{\textnormal{P}_{X}(\textnormal{DIS}(\textnormal{B}(f^*, r)))}{2r} = \frac{\theta(r)}{2},
```

which completes the proof.
\end{proof}

A natural intention and desire is to reduce the probability of disagreement area of version space. It is reasonable to expect some relationship between the number of requested labels and this reduced misclassification yielded by classifiers in version space. On the other hand, having a low expected probability of this set of classifiers implies the performance of chosen classifiers. Theorem \ref{Theorem: vc probability reduction}, characterizes this property. The following proof is inspired by the exact proof that could be found in [hanneke2012activized].
\begin{theorem}[[hanneke2014theory]]

For any \(n \in \mathbb{N}\) and \(r \in (0, 1 )\),
\[
\mathbb{E}\Big[\textnormal{P}_{X}\Big(\textnormal{DIS}\Big(V_{M(n)}^{*}\Big)\Big)\Big] \geq \textnormal{P}_{X}\Big(\textnormal{DIS}\Big(\textnormal{B}(f^{*}, r)\Big)\Big) - nr.
\]

 This also implies that for any \(n \in \mathbb{N}\) and \( \varepsilon \in (0, 1) \),
 \[
 n \leq \theta(\varepsilon)/2 \Longrightarrow\mathbb{E}\Big[\textnormal{P}_{X}\Big(\textnormal{DIS}\Big(V_{M(n)}^{*}\Big)\Big)\Big] \geq \textnormal{P}_{X}(\textnormal{DIS}(\textnormal{B}(f^{*}, r))) /2.
 \]
\end{theorem}

The theorem quantifies the anticipated impact of each label request on reducing the uncertainty associated with the region of disagreement, specifically bounding the expected decrease in probability mass within this region [hanneke2012activized].
This theorem also measures the performance of the algorithm CAL. Investigating fewer unlabeled data points before each `n`-th label request is advantageous, as it enhances algorithmic efficiency by rapidly identifying regions of high disagreement among classifiers within the version space `V^{*}_{M(n)}`. By concentrating on a smaller, more informative subset of data, the algorithm strategically samples areas where label information is most impactful, optimizing label requests while reducing computational and labeling costs.

In sequential learning, the objective is to achieve high accuracy with minimal labeled data, thereby minimizing label complexity. Examining fewer unlabeled points before each label request accelerates the refinement of the version space `V^{*}_{M(n)}` and hastens convergence toward an optimal classifier by effectively shrinking the disagreement region and tightening classifier boundaries.

Conversely, exploring too many unlabeled points can lead the algorithm into regions with minimal disagreement, yielding less informative data and expending unnecessary computational effort. Prompt label requests after investigating fewer unlabeled points ensure that each label significantly contributes to reducing classifier uncertainty and prevents wasting resources on less relevant regions.


\begin{proof}[**Proof of Theorem \ref{Theorem: vc probability reduction**}][hanneke2012activized]The proof proceeds as follows:


\noindent**Step 1: Define Disagreement Region**

For every \(m \in \mathbbm{N}\cup \{0\} \), let \(D_m = {\textnormal{DIS}}({\textnormal{B}}(f,r) \cap V^{*}_{m})\). For simplicity, let \(M(0) = 0\). Explicitly since `\mathcal{C} = V^{*}_0`, thus, we have 
    \( \mathbb{E} \big[ \textnormal{P}_{X}\big(D_{M(0)}\big)\big] = \mathbb{E} \big[ \textnormal{P}_{X}\big(D_{0}\big)\big] = \textnormal{P}_{X}\big({\textnormal{DIS}}({\textnormal{B}}(f,r)\big)\),
    which during the proof we will use it as base case.

    
\noindent**Step 2: Base Case Analysis**

Now, for any fixed \(n \in \mathbbm{N}\) suppose the following term as the inductive hypothesis:
    \[ \mathbb{E} \big[ \textnormal{P}_{X}\big(D_{M(n-1)}\big)\big] \geq \textnormal{P}_{X}\big({\textnormal{DIS}}({\textnormal{B}}(f,r)\big) - (n-1)r.\]

\noindent**Step 3: Establish Probability in Disagreement Region**
 
For any \(x \in D_{M(n-1)}\), according to the definition of disagreement area, there exist \(h_x \in {\textnormal{B}}(f,r) \cap V^{*}_{M(n-1)}\) such that \(
    h_{x}(x) \neq f(x).\) Please note that the \(h_x\) is a random variable that strictly belongs to \(V^{*}_{M(n-1)}\). Now if this \(h_x\) is also a member of \(V^{*}_{M(n)}\), then we can conclude that \(x \in D_{M(n)}\). Formally,
    ```
        \forall x, \mathbf{1}_{D_{M(n)}}(x) &\geq \mathbf{1}_{D_{M(n-1)}}(x) . \mathbf{1}_{V^{*}_{M(n)}}(h_x) \\
        & = \mathbf{1}_{D_{M(n-1)}}(x) . \mathbf{1}_{{\textnormal{DIS}(\{h_x, f\})}^{c}}(X_{M(n)}). 
    ``` 
    
To have \(x \in D_{M(n)}\), two conditions should be satisfied. Namely, this \(x\) should be in the disagreement area of \(V^{*}_{M(n-1)}\) and the related classifier, \(h_x\) also be in the set of the next version space, \(V^{*}_{M(n)}\). Obviously, when \(x \in D_{M(n)}\), the index of \(X\) should be greater or equal to \(M(n)+1\). Because by definition of \(V^{*}_{M(n)}\), for all classifiers in this set, we have \(\forall i\leq M(n), h_x(X_i) = f(X_i)\). On the other hand, investigating points whether they are in \(D_{M(n)}\) or not, naturally implies that the \(D_{M(n-1)}\) ( and clearly \(V^{*}_{M(n-1)}\) ) and  \(V^{*}_{M(n)}\) are available. This also means \(M(n)\) points have already been processed. Therefore, for deciding about the state of any new point, investigation of current and previous version space, namely, \(V^{*}_{M(n-1)}\) and  \(V^{*}_{M(n)}\) is enough. Please note that in the (\ref{induction}) we have the right-hand side equal to or greater than the left-hand side. This inequality can be interpreted as follows: 
```
    ( X_{M(n)+1} \in  D_{M(n-1)} \And h_{X_{M(n)+1}} \in V^{*}_{M(n)} ) \Longrightarrow X_{M(n)+1} \in  D_{M(n)}.
```

It is important to note that in inequality (\ref{induction}), we are restricted only to infer the left-hand side of the inequality from the right-hand side of it. To be more precise, it says that if the right-hand side is one, for sure the left-hand side should be one; otherwise it would be a contradiction. The reason why the other way around is not used is because of the uncertainties and contradictions that come up. Suppose the left-hand side is one  then the following states could happen:
\begin{itemize}
    \item 1 `\geq` 1 `\times` 1 
    \item 1 `\geq` 1 `\times` 0
    \item 1 `\geq` 0 `\times` 1.
\end{itemize}
However, for induction having three possibilities is not going to help. Also, logically based on the established setting, the third case, namely,  ` 1 \geq 0 \times 1`, contextually contradicts the setting. It means that there would be a `x \in D_{M(n)}` and thereof a `h_x \in V^{*}_{M(n)}`, but `x \notin D_{M(n-1)}`. As mentioned before, contextually this is not correct. 
That is, there could be situations where the left-hand side is one but the right-hand side is not. For instance, one equivalent state is when `x` belongs to both `D_{M(n)}` and `D_{M(n-1)}`, but the classier `h_x` that because of it `x` leis in `D_{M(n-1)}` is not a member of `V^{*}_{M(n)}`. In this case, `x` has been lied in `D_{M(n)}`  due to a different classifier than `h_x`. Thus, we can say that both logically and setting-wise this inequality is correct. However, the direction and outcome of this inequality are not practical for induction.


\noindent**Step 4: Calculate Expected Values of Disagreement**
 
Then we will proceed by applying expectations of both sides the (\ref{induction_2}):
\begin{small}
```
        \mathbbm{E}\left[\textnormal{P}_{X}(D_{M(n)})\right]
        &=\mathbbm{E}\left[\mathbf{1}_{D_{M(n)}}(X)\right] \geq 
        \mathbbm{E}\left[ \mathbf{1}_{D_{M(n-1)}}(X) . \mathbf{1}_{{DIS(\{h_x, f\})}^{c}}(X_{M(n)})\right]\notag\\
        &= \mathbbm{E}\left[ \mathbf{1}_{D_{M(n-1)}}(X)\textnormal{P}_{X}\left( h_x(X_{M(n)}) = f(X_{M(n)})\big|X, V_{M(n-1)}^{*}\right) \right]. 
    ``` 

\end{small}
The conditional distribution of `X_{M(n)}` given `V_{M(n-1)}^{*}` is the original distribution `P` but restricted to the set defined by `V_{M(n-1)}^{*}`. This means that while the distribution `P` might be defined over a larger space, we are only interested in the portion of that space where the elements of `V_{M(n-1)}^{*}` disagree.

Since `V_{M(n-1)}^{*}` represents a smaller subset (the disagreement region) where the hypotheses disagree, the distribution needs to be renormalized. Renormalization adjusts the probabilities so that they sum to one within this restricted region, effectively creating a new probability measure conditioned on being in the disagreement region, namely, `\textnormal{P}_{X}(.\big|V_{M(n-1)}^{*})`. Furthermore, since for any `x \in D_{M(n-1)}` the `\textnormal{DIS}\{h_x,f\} \subseteq \textnormal{DIS}(V_{M(n-1)}^{*}) ` is valid, we can say:
\begin{small}
```
    \textnormal{P}_{X}\left( h_x(X_{M(n)}) \neq f(X_{M(n)})\big|V_{M(n-1)}^{*}\right)= \frac{\textnormal{P}_{X}(\textnormal{DIS}(\{h_x,f\}))}{\textnormal{P}_{X}\big(\textnormal{DIS}\big(V_{M(n-1)}^{*}\big)\big)} \leq \frac{r}{\textnormal{P}_{X}(D_{M(n-1)})}. 
```
\end{small}

The inequality arises from the fact that `\textnormal{P}_{X}(\textnormal{DIS}(\{h_x,f\}))` represents the probability that `h_x` and `f` disagree on a randomly chosen data point. Since `h_x \in \textnormal{B}(f,r)`, the maximum probability of disagreement (over all possible `h_x` in this ball) is at most `r`. This is a property of the ball `\textnormal{B}(f,r)`; no hypothesis inside this ball can disagree with `f` more frequently than `r`.
In the following, by plugging in the *complement* of (\ref{complement}) into (\ref{expectation of rhs}), its would be at least:
\begin{small}
    ```
    &\mathbbm{E}\left[\mathbf{1}_{D_{M(n-1)}}(X)\left( 1 - \frac{r}{\textnormal{P}_{X}(D_{M(n-1)})}\right)\right] \notag \\
    &\ \ \ \ \ \ \ \ \ \ \ = \mathbbm{E}\left[\textnormal{P}_{X}\left( X\in D_{M(n-1)}\big| D_{M(n-1)}\right)\left( 1 - \frac{r}{\textnormal{P}_{X}(D_{M(n-1)})}\right)\right]\notag \\
    &\ \ \ \ \ \ \ \ \ \ \ = \mathbbm{E}\left[\textnormal{P}_{X}\left(D_{M(n-1)}\right)\left( 1 - \frac{r}{\textnormal{P}_{X}(D_{M(n-1)})}\right)\right] = \mathbbm{E}\left[\textnormal{P}_{X}\left(D_{M(n-1)}\right) \right] - r. \notag
```
\end{small}

\noindent**Step 5: Conclusion**

Finally, the expansion of the induction proves it:

\begin{small}
```
        \mathbbm{E}\left[\textnormal{P}_{X}(D_{M(n)})\right]
        & \geq \mathbbm{E}\left[\textnormal{P}_{X}\left(D_{M(n-1)}\right) \right] - r \notag\\
        & \geq \mathbbm{E}\left[\textnormal{P}_{X}\left(D_{M(n-2)}\right) \right] - 2r \notag \\
        & ... \notag \\
        & ... \notag \\
        & ... \notag \\
        & \geq \mathbbm{E}\left[\textnormal{P}_{X}\left(D_{M(0)}\right) \right] - nr \notag \\
        &= \textnormal{P}_{X}\left(\textnormal{DIS}(\textnormal{B}(f,r))\right) - nr. 
    ```
By maximizing over `r\geq\varepsilon` and plugging in `n \leq \theta(\varepsilon)/2` into the \ref{P_X-nr},  we will get the implication:
```
  \mathbb{E}\Big[\textnormal{P}_{X}\Big(\textnormal{DIS}\Big(V_{M(n)}^{*}\Big)\Big)\Big] \geq \textnormal{P}_{X}(\textnormal{DIS}(\textnormal{B}(f^{*}, r))) /2.
```
    
\end{small}
\end{proof}

Following are two theorems that based on them the `d`, VC dimension, and disagreement coefficient, `\theta(\varepsilon)` in Theorem \ref{Theorem:label complexity in realizable case} are derived.


\begin{theorem}[[hanneke2014theory]]

If `\mathcal{C}` is the class of k-dimensional linear separators, and `\textnormal{P}_{X}` has density (with respect to `\lambda^{k}`), then:
```
    \forall h \in \mathcal{C}, \theta_{h}(\varepsilon)=o(\frac{1}{\varepsilon}). \notag
```

\end{theorem}

According to this theorem, the `\theta_h(\varepsilon)` should be an asymptotic function of `\frac{1}{\varepsilon}` as in equation \ref{o(v)}. Figure \ref{fig:o(1:epsilon)} shows some candidates for such a function. As we can see in the figure, as `\varepsilon` approaches zero all of the realizations of `o(\frac{1}{\varepsilon})` grow significantly asymptotically slower than `\frac{1}{\varepsilon}`.



The last theorem is about the VC dimension of linear classifiers.

\begin{theorem}[[shalev2014understanding]]
    The VC-dimension of general linear classifiers in `\mathbb{R}^k` is d = k + 1.
\end{theorem}

This theorem highlights the relationship between the dimensionality of the input space and the capacity of linear classifiers to shatter points. Since as in \ref{linear classifier}, we are using linear classifiers with a domain in `[0, 1]^k`, thus:

```
    d = \textnormal{vc}(\mathcal{C})= k+1.
```

## Al Noisy Case

Unlike the realizable case that in its setting, the space of classifiers contains a group of specific and fixed members that at the end, after reducing the non-consistent hypothesis/classifiers, at least one of them would be a desired classifier, in this section we are going to investigate the noisy case where it is based on the relaxation of particular assumptions in realizable cases by injecting noise and distortion into instance space, hypothesis class, and labeling. Furthermore, the output is an estimation of the classifier. Namely, there are no predetermined members among classifiers. This does not mean there would be no frame for the hidden classifier. Some assumptions are taken and would be imposed, but in contrast to realizable scenarios, this method is significantly flexible. Dissimilar to the realizable cases that in some sense impose their classifiers upon the instance space, this noisy framework tries to offer the best representation of the natural configuration of instance space by taking samples from the most uncertain points. In other words, in the end, the algorithm will estimate the optimal classifiers. This is more aligned with real-world problems, where it is less probable that we can fit or even guess the factual classifier(s).


In the following, the setting and definitions are described and then the algorithm and related theorems are investigated.

## Setting
The setting is according to the setting authors have used in [locatelli2017adaptivity].

Consider a feature-label pair `(X, Y)` that follows the joint distribution `\textnormal{P}_{X, Y}`. The marginal distribution of `X`, denoted by `\textnormal{P}_X`, is supported on the unit cube `[0, 1]^k`, while the label `Y` is binary and takes values in `\{0, 1\}`. The conditional distribution of `Y` given `X = x`, denoted as `\textnormal{P}_{Y|X=x}`, is described by the regression function:

`$\eta(x) := \mathbb{E}[Y|X = x], \quad \forall x \in [0, 1]^k.`$
The function `\eta(x)` represents the probability that `Y=1` given that `X=x`. Thus, `\eta(x)` serves as the core function that defines the relationship between `X` and `Y` over the feature space `[0,1]^k`. Additionally, we use `\textnormal{P}` whenever we want to show sampling.

We extend `\eta` to a function `\eta: \mathbb{R}^k \to [0, 1]`, although our main interest is within the unit cube `[0, 1]^k`. The goal of a learning algorithm is to find a classifier `f: [0, 1]^k \to \{0, 1\}` that minimizes the probability of misclassification. The Bayes optimal classifier `f^*` with a classifier `f` that has a minimal excess risk `\mathcal{E}(f)`. Where, 
\[
\mathcal{E}(f) := \mathcal{E}_{\textnormal{P}_{X,Y}}(f) := R(f) - R(f^*).
\]
The excess risk `\mathcal{E}(f)` quantifies the additional error incurred by `f` relative to the optimal Bayes classifier `f^*`. It can also be expressed as:
```
\mathcal{E}(f)=\int_{x \in [0, 1]^k} |1 - 2\eta(x)| \, \mathrm{d}\textnormal{P}_X(x),
```
however, it is clear that over `\{x \in [0, 1]^k : f(x) = f^*(x)\}`, the `R(f) - R(f^*) = 0`. Thus, the excess risk is the result of the region where `\{x \in [0, 1]^k : f(x) \neq f^*(x)\}`. That is:
```
    \mathcal{E}(f)=\int_{x \in [0, 1]^k : f(x) \neq f^*(x)} |1 - 2\eta(x)| \, \mathrm{d}\textnormal{P}_X(x). 
```
This integral indicates that the excess risk depends on the regions where `f` and `f^*` disagree, weighted by `1- 2\eta(x)`. The factor `1- 2\eta(x)` reflects the uncertainty associated with each `x`; points, where `\eta(x)` is close to 0.5, contribute less to the excess risk since the classification decision is less certain.


Furthermore, we will use the symbol `\wedge`, to show the minimum between two numbers. For example:
\[
a \wedge b = \textnormal{min}(a,b),
\]
and the symbol `\vee`, as the logical OR operation. For instance:
\[
 a \vee b = a \ \textnormal{OR}\  b.
\]
Additionally, For any set \( A \), let \( \mathbf{1}_A \) denote the indicator function for \( A \). This function assigns a value of 1 if \( x \in A \) and 0 otherwise, represented by \( \mathbf{1}_A(x) = 1 \) if \( x \in A \), and \( \mathbf{1}_A(x) = 0 \) if \( x \notin A \). Similarly, for any logical condition \( L \), we may use the notation \( \mathbf{1}[L] \), where \( \mathbf{1}[L] = 1 \) when \( L \) is true, and \( \mathbf{1}[L] = 0 \) when \( L \) is false.

The learner can interactively query the label `Y` for any `x \in [0, 1]^k`. The query process follows a Bernoulli trial with a success probability `\eta(x)`, meaning the learner receives `Y=1`, with probability `\eta(x)` and `Y=0`, with probability `1-\eta(x)`. This setup is equivalent to drawing samples from the conditional distribution `\textnormal{P}_{Y|X=x}` for `x \in [0,1]^k` [locatelli2017adaptivity].

Given a fixed budget of `n` samples, `n \in \mathbb{N}`, the learner’s objective is to construct a classifier `\hat{f_n}` that minimizes the excess risk `\mathcal{E}(\hat{f_n})`. The learner must decide which points `x` to sample based on the available budget to best approximate `f^*` with high confidence.

The challenge lies in designing a sampling strategy that chooses the most informative points for querying, such that the resulting classifier `\hat{f_n}` achieves low excess risk `\mathcal{E}(\hat{f_n})` with high probability. This involves selecting points `x` where the label `Y` is uncertain, as these points are likely to yield the most informative updates to the classifier.

## Definitions
In this section, we will have the required definitions.


\begin{definition}\textnormal{{[locatelli2017adaptivity]}}
*[Dyadic grid `G_l`, cells `C`, center `x_C`, and diameter `r_l`]* *A dyadic grid, denoted by \(G_l\), is a regular grid that subdivides the unit cube into \(2^{lk*\) smaller, identical cubes called cells. Each cell has a side length of \(2^{-l}\) and a volume of \(2^{lk}\) . These cells fill the unit cube without overlapping, namely, \([0, 1]^k = \bigcup_{C \in G_l}C\) and \(C \cap C' = \emptyset\) if \(C \neq C'\), with \(C, C' \in G_l\).
The center of a cell \(C\), denoted by \(x_C\), is its geometric center.
The diameter of the cell \(C\) is:
```
    r_l \doteq \max_{x, y \in C} |x - y|_2 = \sqrt{k}2^{-l},
```
where \(|.|_2\) is Euclidean norm.}   
\end{definition}

The Figure(\ref{fig:Dyadic}) shows an example of dyadic grid \(G_l\) with \(k = 3\) for different levels, \(l = [0,1,2]\).



\begin{assumption}\textnormal{{[locatelli2017adaptivity]} (**Strong density**)} 
  There is a definite positive constant \(c_1\) such that for any level \(l \geq 0\) and any cell \(C\) in the grid \(G_l\) with a non-zero probability \(\textnormal{P}_X(C_l)\), it holds:
\[
\textnormal{P}_X(C_l) \geq c_1 k^{k/2} 2^{-lk}.
\]  
\end{assumption}

This assumption ensures that the probability of any cell in the grid is not too small. It allows us to establish a minimum value for the probability of a cell. This assumption holds, for example, when the probability distribution \(\textnormal{P}_X\) is uniform or nearly uniform [locatelli2017adaptivity]. This is why it is an *inequality* and not simply an *equation*.


\begin{definition}\textnormal{{[locatelli2017adaptivity]} (**Hölder smoothness**)} For functions `g: \mathbb{R}^k \rightarrow [0, 1]`, from Hölder class `\Sigma(\alpha,\lambda)` with `\alpha > 0` and `\lambda > 0`, that are ` \lfloor\alpha \rfloor` times continuously differentiable, for any `j \in \mathbb{N}, j \leq \alpha` it holds:

\begin{equation*}
\sup_{x \in \mathbb{R}^k} \sum_{|s| = j} |D^s g(x)| \leq \lambda, \quad *\text{and*}, \quad \sup_{x, y \in \mathbb{R}^k, |x - y| \leq 1} \sum_{|s| = \lfloor\alpha\rfloor} \frac{|D^s g(x) - D^s g(y)|}{|x - y|^{\alpha - \lfloor\alpha\rfloor}_2} \leq \lambda,
\end{equation*}
\textit{
where `D^s f` denotes the mixed partial derivative with parameter `s`. Note that for `\alpha \leq 1` and `\lambda \geq 1`, the condition is reduced to}:
```
\sup_{x, y \in \mathbb{R}^k} \frac{|g(y) - g(x)|}{|y - x|^{\alpha}_2} \leq \lambda. \notag
```


\end{definition}
`\alpha`-Hölder functions possess smoothness properties, facilitating approximation by polynomials of degree `\lfloor\alpha\rfloor` or other approximation schemes, including kernel methods.


\begin{assumption}\textnormal{{[locatelli2017adaptivity]} (**Hölder smoothness** of `\eta`)} Assume `\eta` is a member of the class `\Sigma(\alpha,\lambda)`, defined for positive `\alpha` and non-negative `\lambda`.
\end{assumption}



Figure(\ref{fig:Holder}) illustrates function `g(x) = \sqrt{|x|}` that holds the Hölder smoothness condition for the class `\Sigma(\alpha,\lambda)` with `\alpha = 0.4` and `\lambda = 1`. Figures (\ref{fig:subholder2}) demonstrate the Hölder smoothness in the form, 
\(
|g(x) - g(x_0)| \leq \lambda |x - x_0|^{\alpha}_2,
\) while Figure(\ref{fig:subholder1}) is the extended realization of absolute value:
\[
g(x) \leq g(x_0) + \lambda |x - x_0|^{\alpha}_2,
\]
\[
g(x) \geq g(x_0) - \lambda |x - x_0|^{\alpha}_2.
\]

On the other hand, just by changing `\alpha` and putting it equal to 0.8, the function `g(x)` violates the Hölder smoothness conditions at class `\Sigma(0.8,1)`; Figure(\ref{fig:noHolder}). The intersection of the red and blue lines visualizes the violation of Hölder smoothness's condition. Therefore, we can see that it could happen for some combinations of `\alpha` and `\lambda`, the function `g(x) = \sqrt{|x|}` will not be according to the definition of the Hölder smoothness.


\begin{assumption}\textnormal{{[locatelli2017adaptivity]} (**Margin condition**)}
{There is a non-negative `c_3`, `\Delta_0`, and `\beta` for which `\forall \Delta>0`}:
\begin{small}
```
    \textnormal{P}_{X}(|\eta(X)- 1/2|< \Delta_0)= 0,\  *\text{and*},\   \textnormal{P}_{X}(|\eta(X)- 1/2| \leq \Delta_0 + \Delta) \leq c_3\Delta^{\beta}. \notag
```
\end{small}
\end{assumption}
This assumption essentially describes the behavior of the function `\eta(X)` near the decision boundary, where `\eta(X)` is close to `1/2`. The parameters `\Delta_0, \Delta`, and `\beta` control how fast the probability of being close to the decision boundary decreases as we move away from it.
Since `\Delta_0 \geq 0`, it is obvious that for `\Delta_0 = 0` the `\textnormal{P}_{X}(|\eta(X)- 1/2|< \Delta_0)` would be zero (i.e. distance cannot be negative). Additionally, for `\Delta_0 > 0` having `\textnormal{P}_{X}(|\eta(X)- 1/2|< \Delta_0)` equal to zero implies probabilistically it is impossible  that `\eta(X)` stays in a distance smaller than `\Delta_0` from `1/2`.
Moreover, by increasing the distance from `\Delta_0` to `\Delta_0 + \Delta`, the probability changes. This change will establish an upper bound on the probability. However, consistent with the provided explication about the `\Delta_0`, the probability that the absolute difference between `\eta(X)` and one-half is less than or equal to the sum of `\Delta_0` and `\Delta` is bounded above by *only* `c_3`,`\Delta` and `\beta`; to be more precise, `c_3\Delta^{\beta}`.
It is worth mentioning two well-established instances in literature including *Tsybakov's noise condition* `(\Delta_0 = 0, \beta > 0)` and *Massart's margin condition* `(\Delta_0 = 0, \beta > 0)` that are commonly studied in this field.

\begin{definition}\textnormal{{[locatelli2017adaptivity]}}
{We define `\mathcal{P}(\alpha, \beta, \Delta_0) := \mathcal{P}(\alpha, \beta, \Delta_0; \lambda, c_3)` as the set of classification problems `\textnormal{P}_{X,Y}` characterized by `(\eta; \textnormal{P}_X)`, where Assumptions \ref{Assumption:Hölder smoothness of eta} and \ref{Assumption: Margin condition} hold with parameters `\alpha > 0`, `\beta \geq 0`, and `\Delta_0 \geq 0`, for some fixed `\lambda \geq 1` and `c_3 > 0`. Additionally, we denote `\mathcal{P}^*(\alpha, \beta, \Delta_0)` as the subset of `\mathcal{P}(\alpha, \beta, \Delta_0)` where `\textnormal{P}_X` also satisfies Assumption \ref{Assumption:Strong density}.}
\end{definition}

Please note that the constant values of `c_3>0` and `\lambda \geq 1` are for all subsequent analyses.

\begin{definition}\textnormal{{[locatelli2017adaptivity]} ((`\delta, \Delta, n`)-**correct algorithm**)} Consider a process designed to generate two distinct, measurable subsets, denoted as `S^0, S^1`, within the k-dimensional unit cube, `[0,1]^k`.
 Let `0 < \delta < 1`, and `\Delta \geq 0`. We refer to such a procedure as **weakly** `(\delta, \Delta, n)`-**correct** for a classification problem `\textnormal{P}_{X,Y}` (characterized by `(\eta, \textnormal{P}_X)`) provided that, with a probability exceeding `1 - 8\delta`  across a maximum of `n` label inquiries:
\[
\left\{x \in [0,1]^k : \eta(x) - 1/2 > \Delta \right\} \subset S^1, \text{ and } \left\{x \in [0,1]^k : 1/2 - \eta(x) > \Delta \right\} \subset S^0.
\]
Furthermore, assuming the occurrence of the same probabilistic event across a maximum of n label requests, we observe that:
\[
S^1 \subset \left\{x \in [0,1]^k : \eta(x) - 1/2 > 0 \right\}, \text{ and } S^0 \subset \left\{x \in [0,1]^k : \eta(x) - 1/2 < 0 \right\}.
\]
Consequently, such a procedure is conventionally termed `(\delta, \Delta, n)`-**correct** for `\textnormal{P}_{X,Y}`.

\end{definition}
In the next section, an algorithm and the theory behind it will be discussed. 
\newpage

## Algorithm and Theorems
We start with the following algorithm which is a modification of the original version in [locatelli2017adaptivity]. Since, in this work, we are supposed to focus on the case where `\alpha \leq 1`, the algorithm has been modified accordingly. Thus, wherever in the original work at [locatelli2017adaptivity], there was ` \alpha \wedge 1` or `\alpha \vee 1`, we would replace them accordingly by `\alpha` and `1`.
\begin{algorithm}[H] 
\caption{}
\begin{algorithmic}[1]
\State **Input:** `n, \delta, \alpha, \lambda`
\State **Initialisation:** `t = 2^k t_{1, \alpha},\ l = 1, \ \mathcal{A}_1 = G_1 \text{(active space)}, \ \forall l' > 1,\  \mathcal{A}_{l'} = \emptyset, \ S^0 = S^1 = \emptyset`
\While{`t \leq n`}
    \For{each active cell `C \in A_l`}
        \State Request `t_{l, \alpha}` samples `(\tilde{Y}_{C,i})_{i \leq t_{l, \alpha}}` at the center `x_C` of `C`
        \If{`|\hat{\eta}(x_C) - 1/2| \leq B_{l,\alpha}`}
            \State `\mathcal{A}_{l+1} = \mathcal{A}_{l+1} \cup \{C' \in G_{l+1} : C' \subset C\}` \begin{small}
                \Comment{keep all children `C'` of `C` active}
            \end{small}
        \Else
            \State Let `y = 1\{\hat{\eta}(x_C) \geq 1/2\}`
            \State `S^y = S^y \cup C` \Comment{label the cell as class `y`}
        \EndIf
    \EndFor
    \State Increase depth to `l = l + 1`, and set `t = t + |\mathcal{A}_l| \cdot t_{l, \alpha}`
\EndWhile
% \State Set `L = l - 1`
% \If{`\alpha > 1`}\Comment{Not our case}
%     \State ` \text{Run related Algorithm}` 
% \EndIf
\State **Output:** `S^y` for `y \in \{0, 1\}`, and `\hat{f}_{n,\alpha} = \mathbf{1}\{S^1\}`
\end{algorithmic}
\end{algorithm}
The algorithm presented is designed given the \(\lambda\) and \(\alpha\), to focus its exploration of the data space specifically on regions where classifying the problem is most challenging. This difficulty arises particularly in areas where the function \(\eta\), hovers near the critical threshold of \(1/2\), making it hard to classify. The algorithm achieves this targeted exploration by incrementally refining a partition of the space. This partitioning is structured based on a dyadic tree, which is for recursively splitting the space into smaller and smaller sections.

At each level of depth \(l\) within this partitioning tree, the algorithm selects a specific point, \(x_C\), which is the center of an active cell 
\(C\) within the partition \(\mathcal{A}_l\). The algorithm then samples this point \(x_C\) a predetermined number of times, denoted as \(t_{l, \alpha} \) where:
``` 
t_{l, \alpha} =  \frac{\log_2(1/\delta_{l,\alpha})}{2b_{l,\alpha}^2}, \ \ \ \text{if } \alpha \leq 1.
```
Specifically, \(b_{l,\alpha}\) is given by \(\lambda k^{ (\alpha)/2}2^{-l\alpha}\), while \(\delta_{l,\alpha}\) scales the \(\delta 2^{-l(k+1)}\), where \(\delta\) is a predefined parameter that controls the confidence level of the classification.

The core of the algorithm's decision-making process lies in estimating the function \(\eta(x_C)\) at the point \(x_C\) using the collected samples.  This estimate, \(\hat{\eta}(x_C)\), , is computed as the average of the sample values \(\Tilde{Y}_{C,i}\) at \(x_C\) normalized by the number of samples \(t_{l, {1}}\), that is:
```
    \hat{\eta}(x_C) = t_{l, { \alpha}}^{-1} \sum_{i= 1}^{t_{l, { \alpha}}}\Tilde{Y}_{C,i}. \notag
```

The next step in the algorithm involves comparing the absolute difference between this estimate \(\hat{\eta}(x_C)\) and the critical threshold \(1/2\). If this difference, \(|\hat{\eta}(x_C) - 1/2|\),  is sufficiently large, relative to a total error comprising bias and deviation bound:
```
    B_{l,\alpha} = 2 \left[ \sqrt{\frac{\log_2(1/\delta_{l, \alpha})}{2t_{l, \alpha}}} + b_{l, \alpha} \right], 
```
then the algorithm can confidently label the cell \(C\) as belonging to the class \(S^0\) or \(S^1\). This labeling process is crucial for the algorithm to decide the class of cells based on the empirical evidence:
```
    \mathbf{1}\big\{\hat{\eta}(x_C) \geq 1/2 \big\}. \notag
```

However, if the difference \(|\hat{\eta}(x_C) - 1/2|\) is too small, indicating that the algorithm is not confident enough in the classification, the partition is further refined. The cell \(C\) is divided into smaller subcells, and these new, finer cells become the active cells at the next depth level \(l+1\).

The process of refining and labeling continues until the algorithm reaches a point where the total used budget—measured in terms of the number of samples taken— that is \(t +t_{l, {\alpha}}.|\mathcal{A}_l| \) exceeds the allowed limit \(n\). When this happens, the algorithm stops, and the final depth reached is denoted as \(L\).

The Equation \ref{deviation bound}, which makes a confidence bound 
around `\eta(x)`, consists of two main components. The deviation of the unknown optimal classifier, `\sqrt{\frac{\log(1/\delta_{l, \alpha})}{2t_{l, \alpha}}}`, plus the deviation of its estimation, `b_{l, \alpha}`. Adding them up and multiplying them by two will establish `B_{l, \alpha}` a significantly conservative bound to decide about the cell's destiny.



\begin{theorem}[[locatelli2017adaptivity]]

Algorithm \ref{Alg:Non-adaptive Subroutine} run on a problem \(\mathcal{P}^{*}(\lambda, \alpha, \beta , \Delta_0)\) with input parameters \(n, \delta, \alpha, \lambda\) is \( (\delta, \Delta^{*}_{n, \delta, \alpha, \lambda}, n)\)-correct, with:
```
    \Delta^{*}_{n, \delta, \alpha, \lambda} =
12\sqrt{k}\bigg( \frac{c_7 \lambda^{(\frac{k}{\alpha}\vee \beta)}\textnormal{log}\big(\frac{2d\lambda^{2}n}{\delta} \big)}
{(2\alpha+[k -\alpha\beta]_{+})\alpha\ n} \bigg)^{\frac{\alpha}{ 2\alpha+[k -\alpha\beta]_{+}}} & , for\  \alpha \leq 1,\notag \\ \notag
```
with \(c_7 = 2(k+1)c_5\) and \(c_5 = 2^{ \beta}\textnormal{max}\big(\frac{c_3}{c_1}8^{\beta}, 1 \big)\), where \(c_1\) and \(c_3\) are the constants involved in Assumption \ref{Assumption:Strong density} and \ref{Assumption: Margin condition} respectively.
\end{theorem}

The proof of Theorem \ref{theorem: non-adaptive}, depends on several lemmas and consists of six steps. The lemmas will be stated step by step. The proof of lemmas will be declared separately at the end after the proof of the Theorem \ref{theorem: non-adaptive}. This helps avoid confusion.

\begin{proof}[**Proof of Theorem \ref{theorem: non-adaptive**}][locatelli2017adaptivity]
    We will use the following abbreviation to make life easier:
    \[
    t_l = t_{l, 1},\  b_l = b_{l, \alpha},\  \delta_l = \delta_{l,1},\  B_l = B_{l, 1},\  \textnormal{and } N_l = |\mathcal{A}_l|.
    \]
\noindent **Step 1: A favorable event.**

\noindent Let \( C \) be a cell at depth `l`. The following event is defined:
``` 
\xi_{C,l} = \left\{
\left| t_l^{-1} \sum_{u=1}^{t_l} \mathbf{1}(\tilde{Y}_{C,i} = 1) - \eta(x_C) \right| \leq \sqrt{\frac{\log(1/\delta_l)}{2t_l}} 
\right\},
```
such that the \( (\tilde{Y}_{C,i})_{i \leq t_l} \) are samples obtained within cell `C` at point `x_C` if the algorithm chooses to sample in cell `C`. Recall that:
\[
\hat{\eta}(x_C) = t_l^{-1} \sum_{i=1}^{t_l} \mathbf{1}(\tilde{Y}_{C,i} = 1).
\]
We define the event \( \xi \) as follows:
\[
\xi = \left\{ \bigcap_{l \in \mathbb{N}^*, C \in {G}_l} \xi_{C,l} \right\}.
\]
The left-hand side of inequality in the above set (\ref{def: xi_Cl}) comes from Hoeffding’s inequality for estimating *confidence intervals*. We know that `\eta(x_C)` is a function between zero and one, so:
```
        &\textnormal{P}\left(\hat{\eta}(x_C) \notin \left[\eta(x_C) - \varepsilon, \eta(x_C) + \varepsilon  \right] \right) \geq 2\text{exp}(-2\varepsilon^2t_l), \notag\\ \notag
```
which is equivalent to:
```
        &\textnormal{P}\left(\left| \hat{\eta}(x_C) - \eta(x_C)\right| > \varepsilon \right) \geq 2\text{exp}(-2\varepsilon^2t_l).\\ \notag
```
Now, let's put ` \delta_l = \text{exp}(-2\varepsilon^2t_l)`. We will have `\varepsilon` as following:
``` 
        & \frac{1}{\delta_l} = \text{exp}(2\varepsilon^2t_l),\\ 
        & \varepsilon = \sqrt{ \frac{\text{ln}(\frac{1}{\delta_l})}{2t_l}} . \notag\\ \notag
```
Since for the given ` 0< \delta_l < 1`, it always holds that `\text{ln}(\frac{1}{\delta_l}) \geq \text{log}((\frac{1}{\delta_l}))`, thus:
``` 
        & \sqrt{ \frac{\text{log}(\frac{1}{\delta_l})}{2t_l}} \leq \varepsilon.\\\notag
```
Finally by plugging in (\ref{1/delta_l}) and (\ref{log(1/delta_l)}) into (\ref{Hoeffding}), we get the following inequality that later will be used for proving other lemmas:
``` 
        &\textnormal{P}\left(\left| \hat{\eta}(x_C) - \eta(x_C)\right| > \sqrt{ \frac{\text{log}(\frac{1}{\delta_l})}{2t_l}} \right) \geq 2\delta_l.\\ \notag
```

\begin{lemma}\textnormal{[locatelli2017adaptivity]} {It follows that
\[
\textnormal{P}(\xi) \geq 1 - 4\delta.
\]
Furthermore, on the event \( \xi \), we have}
```
   |\widehat{\eta}(x_C) - \eta(x_C)| \leq b_l.

```

\end{lemma}
\noindent**Step 2: No mistakes on labeled cells.**

\noindent For any integer \( l \in \mathbb{N}^* \), let `C` be an element of `G_l`. We can express:
```
    \widehat{k}^*_C = \mathbf{1}\{\widehat{\eta}(x_C) \geq 1/2\} \quad \text{additionally, let us define} \quad k^*_C \doteq \mathbf{1}\{\eta(x_C) \geq 1/2\}. \notag
```
\begin{lemma}\textnormal{[locatelli2017adaptivity]}{We have that on} \( \xi \),
```
    \forall y \in \{0,1\}, \forall C \in S^y, \forall x \in C, \quad \mathbf{1}\{\eta(x) \geq 1/2\} = y. 
```

*This implies that:*
```
    S^1 \subset \{x : \eta(x) - 1/2 > 0\} \quad *and,* \quad S^0 \subset \{x : \eta(x) - 1/2 < 0\}. \notag
```
\end{lemma}

\noindent**Step 3: Maximum gap with respect to 1/2 for all active cells**

\noindent When cell `C` is split and incorporated into `\mathcal{A}_{l+1}` at depth `l \in \mathbb{N}^*`, the Algorithm (\ref{Alg:Non-adaptive Subroutine}) and the definition of `\xi`, as outlined in Equation (\ref{lem:lemma 1}), and by the means of properties of triangle inequalities ensure that:
 
```
    |\eta(x_C) - 1/2| - |\widehat{\eta}(x_C) - 1/2| \notag
    & \leq |\eta(x_C) - \widehat{\eta}(x_C)|\\ \notag
    & \leq |\eta(x_C) - 1/2| + |\widehat{\eta}(x_C) - 1/2| \leq 2b_l,\\ \notag
```
which simply gives us the upper bound for the left-hand side of it as:
```
|\eta(x_C) - 1/2| - |\widehat{\eta}(x_C) - 1/2| \leq 2b_l, 
```
and similarly:
```
    |\widehat{\eta}(x_C) - 1/2| - |\eta(x_C) - 1/2| \notag
    & \leq |\eta(x_C) - \widehat{\eta}(x_C)| \notag\\
    & \leq |\eta(x_C) - 1/2| + |\widehat{\eta}(x_C) - 1/2| \leq 2b_l,\\ \notag
```
yields following upper bound too:
```
    |\widehat{\eta}(x_C) - 1/2| \notag
    & \leq |\eta(x_C) - \widehat{\eta}(x_C)| + |\eta(x_C) - 1/2| \notag\\
    & \leq 2|\eta(x_C) - 1/2| + |\widehat{\eta}(x_C) - 1/2| \leq 3b_l,\\ \notag
```
or simply:
```
    |\widehat{\eta}(x_C) - 1/2| \leq 3b_l. 
```
Finally combining (\ref{upper bound for triangles}) and (\ref{3b_l upper bound}) results in:
```
    |\eta(x_C) - 1/2| - b_l \leq |\widehat{\eta}(x_C) - 1/2| \leq 4b_l,
```
which suggests `|\eta(x_C) - 1/2| \leq 5b_l`. Based on Equation (\ref{|eta(x) - eta(y)| < b_l}), we can conclude that for any cell `C` scheduled to be split and added to `\mathcal{A}_{l+1}` and for any element `x \in C`, the following condition holds on `\xi`:
```
    |\eta(x) - 1/2| - b_l \leq |\eta(y) - 1/2| \leq 5b_l, \notag
```
plainly:
```
    |\eta(x) - 1/2| \leq 6b_l \doteq \Delta_l. 
```
\noindent**Step 4: Bound on the number of active cells**

\noindent For any `\Delta \geq 0`, we define the set:
\[
\Omega_\Delta = \{x \in [0, 1]^k : |\eta(x) - 1/2| \leq \Delta\},
\]

and any positive integer `l`, let `N_l(\Delta)` denote the count of cells `C` belonging to the set `G_l` that are completely contained within the region `\Omega_\Delta`, namely, `C \subset \Omega_\Delta`.

\begin{lemma}\textnormal{[locatelli2017adaptivity]}{On the event `\xi`, we have:}

```
    N_{l+1} &\leq \frac{c_3}{c_1}[\Delta_l - \Delta_0]_+^{\beta} r_{l+1}^{-k} \notag \\
    & \leq c_5 \lambda^{\beta} r^{-[k-(1)\beta]_{+}}_{l+1}\mathbf{1}{\{\Delta_l > \Delta_{0}\}}.\\ \notag
```
\end{lemma}



This lemma is significant in bounding the complexity of the active learning process. By controlling the number of active cells, one can manage the sample complexity and ensure the learning process remains efficient. The lemma essentially says that as the resolution increases (as `l` increases), the number of cells that are still "active" and entirely within the uncertain region `\Omega` can be bounded in terms of the size of the cells and the noise parameter `\Delta`.


\noindent**Step 5: A minimum depth.**

\begin{lemma}\textnormal{[locatelli2017adaptivity]}{On the event `\xi`, we obtain the following results regarding `L`:}

*since *`\alpha \leq 1`* we have the following*
```
  L \geq \frac{1}{2\alpha + [k - \alpha\beta]_+} \log_2 \Big( \frac{(2\alpha + [k - \alpha\beta]_+)2\alpha n}{c_7 \lambda^{\beta-2} \log(\frac{2k\lambda^2 n}{\delta})} \Big) - 1,  
```

*with `c_7 = 2c_5(k+1)`. Alternatively, the algorithm may terminate before reaching depth `L`, resulting in `\mathcal{E*(\hat{f_n}) = 0`.}

\end{lemma}

\noindent**Step 6: Conclusion.**

\noindent From this point on, we write `S^0, S^1` for the sets that Algorithm (\ref{Alg:Non-adaptive Subroutine}) outputs at the end (so the sets at
the end of the algorithm).

We present the following lemma.


\begin{lemma}\textnormal{[locatelli2017adaptivity]}{If \(S^1\) and \(S^0\) are disjoint sets, \(S^1 \cap S^0 = \emptyset\), and if there exists a non-negative value \(\Delta_L \geq 0\),  such that on some event \(\xi'\), we have}
\[
\{x \in [0,1]^k: \eta(x) - \Delta_L \geq 1/2\} \subset S^1, \text{ and } \{x \in [0,1]^k: \eta(x) + \Delta_L \leq 1/2\} \subset S^0,
\]
Then, on the event `\xi'`, it follows that
\[
\sup_{x \in [0,1]^k: \hat{f}_{n,a} \neq f^*(x)} |\eta(x) - 1/2| \leq \Delta_L, \text{ and } \textnormal{P}_{X}(\hat{f}_{n,a} \neq f^*) \leq c_3 \Delta_{L}^{\beta}\mathbf{1} \{\Delta_L \geq \Delta_0\},
\]
\text{ and }
\[
\mathcal{E}(\hat{f}_{n,a}) \leq c_3 \Delta_{L}^{1+\beta} \mathbf{1}\{ \Delta_L \geq \Delta_0 \}.
\]


\end{lemma}

Finally, the above Lemmas gives us the requirements to prove the Theorem. Observe that by definition of the algorithm, `S^1 \cap S^0 = \emptyset`, which means they are disjoint sets. According to Equations (\ref{lemma2 part one}), (\ref{step3 last ineq.})  we know that on the event `\xi` (and consequently with a probability exceeding `1 - 4\delta`):
\begin{small}
```
   \{x \in [0, 1]^k: \eta(x) - \Delta_L \geq 1/2\} \subset S^1, \text{ and } \{x \in [0,1]^k: \eta(x) + \Delta_L \leq 1/2\} \subset S^0,  
```
\end{small}

where
```
    \Delta_L & \leq 6 b_{l, \alpha}\\ \notag
            & = 6 \lambda k^{\frac{\alpha}{2}}2^{-l\alpha}.  \\ \notag
```
thus by manipulating Equations \ref{lemma4}
```
\Delta_L &\leq 6\lambda{k}^{\alpha/2}2^{\alpha} \Big(\frac{c_7 \lambda^{\beta-2} \log \left(\frac{2k\lambda^2 n}{\delta}\right)}{(2\alpha + [k - \alpha\beta]_+) \cdot 2\alpha n} \Big)^{\alpha/(2\alpha + [k - \alpha\beta]_+)} \notag \\
&\leq 12\lambda k^{\alpha/2} \Big(\frac{c_7 \lambda^{\beta-2} \log \left(\frac{2d\lambda^2 n}{\delta}\right)}{(2\alpha + [k - \alpha\beta]_+) \cdot 2\alpha n} \Big)^{\alpha/(2\alpha + [k - \alpha\beta]_+)} \notag \\
&\leq 12\sqrt{k}\Big(\frac{c_7 \lambda^{\beta-2} \log \left(\frac{2k\lambda^2 n}{\delta}\right)}{(2\alpha + [k - \alpha\beta]_+) \cdot 2\alpha n} \Big)^{\alpha/(2\alpha + [k - \alpha\beta]_+)}. \notag
```

According to Equation (\ref{lemma4}), this implies the validity of Theorem \ref{theorem: non-adaptive} for `\alpha \leq 1`. Consequently, by Lemma (5), we have on the event `\xi` (and therefore with a probability exceeding `1 - 4\delta`):
```
\sup_{x \in [0,1]^k: \hat{f}_{n,a} \neq f^*(x)} |\eta(x) - 1/2| &\leq \Delta_L \notag\\
&\leq 12\sqrt{k}\Big(\frac{c_7 \lambda^{\beta-2} \log \left(\frac{2k\lambda^2 n}{\delta}\right)}{(2\alpha + [k - \alpha\beta]_+) \cdot 2\alpha n} \Big)^{\alpha/(2\alpha + [k - \alpha\beta]_+)}.\notag
```
Furthermore,
```
\textnormal{P}_{X}(\hat{f}_{n,a} \neq f^*(x)) &\leq c_3 \Delta_{L}^{\beta} \mathbf{1} \{\Delta_L \geq \Delta_0\}\notag\\
&\leq c_3 12^{\beta}\sqrt{k}\Big(\frac{c_7 \lambda^{({\frac{k}{\alpha}}\vee\beta)} \log \left(\frac{2k\lambda^2 n}{\delta}\right)}{(2\alpha + [k - \alpha\beta]_+) \cdot 2\alpha n} \Big)^{\alpha\beta/(2\alpha + [k - \alpha\beta]_+)} \notag
```
and
```
\mathcal{E}(\hat{f}_{n,a}) &\leq c_3\Delta_L^{\beta+1} \mathbf{1}\{\Delta_L \geq \Delta_0\}\notag\\
&\leq c_3 12^{\beta+1}\sqrt{k}\Big(\frac{c_7 \lambda^{({\frac{k}{\alpha}}\vee\beta)} \log \left(\frac{2k\lambda^2 n}{\delta}\right)}{(2\alpha + [k - \alpha\beta]_+) \cdot 2\alpha n} \Big)^{\alpha(\beta+1)/(2\alpha + [k - \alpha\beta]_+)}. \notag
```
\end{proof}
%%%%%%%%%%%
% \vspace

In this section, the proofs of the Lemmas are given. They are somehow similar to proofs in [locatelli2017adaptivity]. However, more details and explanations are given wherever it could help for more clarification and understanding.


\begin{proof} [**Proof of Lemma \ref{lemma_noise: Lemma_1**}][locatelli2017adaptivity]
According to Hoeffding's inequality, and definition of `\xi_{C,l}` in (\ref{def: xi_Cl}) we know that:
```
    \textnormal{P}(\xi_{C,l}) = \textnormal{P}\left(\left| t_l^{-1} \sum_{u=1}^{t_l} \mathbf{1}(\tilde{Y}_{C,i} = 1) - \eta(x_C) \right| \leq \sqrt{\frac{\log(1/\delta_l)}{2t_l}} \right)
    \geq 1 - 2\delta_l.\notag
```

And if for its complement, `\xi_{C,l}^c`:
```
    \textnormal{P}(\xi_{C,l}^c) = 1-\textnormal{P}(\xi_{C,l}) \leq 2\delta_l, \notag
```
which is equivalent to:
```
    \textnormal{P}(\xi_{C,l}^c) = 1-\textnormal{P}(\xi_{C,l}) \leq 2\delta_l. \notag
```
Let us now turn our attention to,
\[
\xi = \left\{\bigcap_{l \in \mathbb{N}^*, C \in G_l} \xi_{C,l}\right\},
\]
We examine the intersection of events defined by the condition that, for every depth `l` and every cell `C \in G_l`, the preceding event occurs. Given that there are `2^{ld}` such events at depth `l`, a straightforward application of the union bound yields:
``` 
   \textnormal{P}\left(\xi \right) = \textnormal{P}\left(\bigcap_{l \in \mathbb{N}^*, C \in G_l} \xi_{C,l} \right) \notag & = 1- \textnormal{P}\left(\bigcup_{l \in \mathbb{N}^*, C \in G_l} \xi_{C,l}^{c} \right) \notag \\
    & \geq 1 - \sum_{l} 2\delta_l 2^{lk} \notag\\
    & = 1 - \sum_{l} 2\delta2^{-l(k+1)} 2^{lk} \notag\\
    & = 1 - 2\delta \sum_{l} 2^{-l} \notag\\
    & = 1 - 2\delta \left( \frac{1}{1- \frac{1}{2}}\right) \notag\\
    & = 1 - 4\delta,\notag\\ \notag
```
or simply, `\textnormal{P}(\xi) \geq 1 - \sum_{l=1} 2^{lk} \delta_i \geq 1 - 4\delta` as we have set `\delta_l = \delta{2}^{-l(k+1)}`.
We define `b_l = \lambda k^{1/2} 2^{-l)}` for any `l \in \mathbb{N}^*`. Due to Defination (\ref{Def:Hölder smoothness}) and Assumption (\ref{Assumption:Hölder smoothness of eta}), given any `x, y \in C`, where `C \in G_l`, it follows that:
```
    |\eta(x) - \eta(y)| \leq \lambda |x - y|_{2}^{\alpha} \leq \lambda k^{\frac{1}{2}}2^{-l} = b_l, 
```
this is because the farthest distance between two points in the cell `C` is at most equal to the diameter of the cell, namely:
\[
r_l \doteq \max_{x, y \in C} |x - y|_2 = {k^{\frac{1}{2}}}2^{-l}.
\]

Given event `\xi`, for all `l \in \mathbb{N}^*`, as we have defined `t_l = \log(1/\delta_l)`, substituting this value into the bound yields the following inequality at time `t_l` for every cell `C \in G_l`:

\[
|\widehat{\eta}(x_C) - \eta(x_C)| \leq b_l.
\]
\end{proof}

\begin{proof} [**Proof of Lemma \ref{lemma_noise: Lemma_2**}][locatelli2017adaptivity]
Using Equations \ref{lem:lemma 1} and \ref{|eta(x) - eta(y)| < b_l}, we derive:

\[
4b_l < \widehat{\eta}_{\widehat{k}_C^*}(x_C) - \frac{1}{2} < \eta_{\widehat{k}_C^*}(x_C) + b_l - \frac{1}{2},
\]

which indicates that \( \eta_{\widehat{k}_C^*}(x_C) - \frac{1}{2} > 3b_l > 0 \). Therefore, according to the definition of \( k_C^* \), it follows that \( \widehat{k}_C^* = k_C^* \).

Additionally, under the smoothness condition, for any \( x \in C \), we have:

\[
|\eta(x) - \eta(x_C)| \leq \lambda k^{1/2} |x - x_C| \leq b_l.
\]

Assuming without loss of generality that \( \widehat{k}_C^* = 1 \), we obtain from the previous result that \( \widehat{k}_C^* = k_C^* = 1 \) and \( \eta(x_C) - \frac{1}{2} > 2b_l \). Thus, for \( x \in C \), we have \( \eta_{k_C^*}(x) - \frac{1}{2} > 0 \), confirming that \( k_C^* \) is indeed the optimal class within the cell \( C \). The labeling \( \hat{k}_C^* = k_C^* \) aligns with the Bayes classifier across the entire cell, ensuring no additional risk within \( C \).

Summarizing, on \( \xi \), we have:

\[
\forall y \in \{0, 1\}, \forall C \in S^y, \forall x \in C, \quad 1\{\eta(x) \geq \frac{1}{2}\} = y.
\]

Consequently, this implies that:

\[
S^1 \subset \{x : \eta(x) - \frac{1}{2} > 0\} \quad \text{and} \quad S^0 \subset \{x : \eta(x) - \frac{1}{2} < 0\}.
\]

\end{proof}
\begin{proof} [**Proof of Lemma \ref{lemma_noise: Lemma_3**}][locatelli2017adaptivity] By Assumption (\ref{Assumption: Margin condition}), `\textnormal{P}_X(\Omega)\leq c_3[\Delta - \Delta_0]_+^{\beta}\mathbf{1}\{\Delta \geq \Delta_0\}` exists. Thus, we can infer from Assumption (\ref{Assumption:Strong density}) that:
```
 N_l(\Delta)\leq \frac{c_3}{c_1}[\Delta - \Delta_0]_+^{\beta} r_{l}^{-k}\mathbf{1}\{\Delta \geq \Delta_0\}.    
```
On the other hand from Definition (\ref{Def:Hölder smoothness}), Hölder Smoothness, where `\alpha \leq 1` we know that:
```
    [\Delta_l - \Delta_0]_+^{\beta} \leq \lambda^{\beta}r_{l+1}^{\beta}. 
```
Plugging in (\ref{d_l-d_0 <lambda r}) into (\ref{proof of lemma 3 [N_l(Delta)]}) will result in:
```
    N_{l+1}(\Delta_l)
    & \leq \frac{c_3}{c_1}[\Delta_l - \Delta_0]_+^{\beta} r_{l}^{-k}{\mathbf{1}}\{\Delta_l \geq \Delta_0\} \notag \\
    & \leq \frac{c_3}{c_1}\lambda^{\beta} r_{l+1}^{\beta-k}{\mathbf{1}}\{\Delta_l \geq \Delta_0\} \notag\\
    & \leq c_5 \lambda^{\beta} r_{l+1}^{-[k-\beta]}{\mathbf{1}}\{\Delta_l \geq \Delta_0\}, 
```
where `N_{l+1}` represents the count of active cells at the commencement of the depth `(l + 1)` round, and `[.]_+ = max(.,0)` and `c_5 = max(\frac{c_3}{c_1}, 1)`. This formula holds true for `L-1 \geq l \geq 0.`

\end{proof}

\begin{proof} [**Proof of Lemma \ref{lemma_noise: Lemma_4**}][locatelli2017adaptivity] For each depth level `1 \leq l \leq L`, we sample these active cells a total of `t_l=\frac{\log(1/\delta_{l})}{2b_{l}^2}` times. Initially, let's examine the case where `\Delta_0= 0`. To establish an upper bound on the total number of samples needed by the algorithm to attain depth `L`, we can refer to Equation (\ref{reslut of lemma 3  for[N_l(Delta)]}), which states that on the event `\xi`:

```
\sum_{l=1}^{L} N_{l}t_{l} + N_{L}t_{L} &\leq 2 \left(\sum_{i=1}^{L} \left(c_5 \lambda^{\beta} r^{- [k - \alpha \beta]_{+}}\right) \frac{\log(1/\delta_{l})}{2\lambda^{2} r^{2\alpha}_{l}} \right) \notag \\
&\leq 2c_5 \lambda^{\beta - 2} \log(1/\delta_L) \sum_{l=1}^{L} r_{l}^{-(2\alpha + [k - \alpha\beta]_{+})} \notag \\
&\leq 2c_5 k^{-(2 \alpha + k - \alpha \beta)/2} \lambda^{\beta - 2} \log(1/\delta_L) \frac{2^{L(2\alpha + [k - \alpha\beta]_{+})}}{2^{(2\alpha + [k - \alpha\beta]_{+})} - 1} \notag\\
&\leq \frac{4c_5}{k^{(2 \alpha + k - \alpha \beta)/2}} \lambda^{\beta - 2} \log(1/\delta_L) \frac{2^{L(2\alpha + [k - \alpha\beta]_{+})}}{2\alpha + [k -\alpha\beta]_{+}}. \notag
```
Since `2a - 1 \geq a/2` for any positive real number, `a \in \mathbb{R}^{+}`, we can conclude that on the event `\xi`:
```
\sum_{l=1}^{L} N_{l}t_{l} + N_{L}t_{L} \leq {4c_5} \lambda^{\beta - 2} \log(1/\delta_L) \frac{2^{L(2\alpha + [k - \alpha\beta]_{+})}}{2\alpha + [k -\alpha\beta]_{+}}. 
```
To establish an upper bound for `L`, we'll employ a straightforward approach. Since `t_L`, the number of samples at depth `L` must be less than `n` (if there were only one active cell, which represents the minimum possible number of active cells, the allocated budget would be insufficient), we can deduce the following:

```
    \frac{\log(1/\delta_{L})}{2\lambda^{2} r^{2\alpha}_{L}} \leq n, \notag
```
this directly implies, given that `\delta_L <\delta < e^{-1}`, that:
```
    L \leq \frac{1}{2\alpha} \log_2(2k\lambda^{2}n). \notag
```
We can now determine an upper bound for `\log(1/\delta_L)`:
```
    \log(1/\delta_L) = \log(2^{L(k+1)}/\delta)
    & \leq \frac{k+1}{2\alpha}\log_2(2k\lambda^{2}n) + \log(1/\delta) \notag\\
    & \leq \frac{k+1}{2\alpha}\log_2(\frac{2k\lambda^{2}n}{\delta}). 
```
By combining Equations (\ref{lemma 4 bound for log(1/delta)}) and (\ref{ Eq. in lemma 4}), we can infer that on the event `\xi`, the allocated budget is sufficient to achieve the target depth:
```
    L \geq \left\lfloor \frac{1}{2\alpha + [k -\alpha\beta]_{+}} \log_{2} \left( \frac{(2\alpha + [k -\alpha\beta]_{+})2\alpha n}{c_7 \lambda^{\beta -2} \log_2\left(\frac{2k\lambda^{2}n}{\delta}\right)}\right)
    \right\rfloor,  \notag
```
where `c_7 = 2c_5(k + 1)`. Alternatively, the algorithm may terminate before reaching depth `L`, with `S^1 \cup S^0 = [0,1]^k`, leading to a zero extra risk.

\end{proof}


\begin{proof} [**Proof of Lemma \ref{lemma_noise: Lemma_5**}][locatelli2017adaptivity] The lemma's assumption directly implies the first conclusion. The second conclusion is a direct corollary of the lemma's hypothesis and Assumption (\ref{Assumption: Margin condition}). In other words, we know that:
```
    \textnormal{P}_{X}(\hat{f}_{n,a} \neq f^*) = \int_{x \in [0, 1]^k} \mathbf{1}\{\hat{f}(x) \neq f^*(x)\} \, \mathrm{d}\textnormal{P}_X(x).
```
However, instead of calculating this difficult integral, we can find an upper bound for it. This can happen through the Assumption (\ref{Assumption: Margin condition}):

For the third conclusion, we have the following integral that again we prefer to find an upper bound for it with the worst-case scenario: 

```
    \mathcal{E}(\hat{f}_{n,a})
    &=\int_{x \in [0, 1]^k : \hat{f}_{n,a} \neq f^*(x)} |1 - 2\eta(x)| \ \mathrm{d}\textnormal{P}_X(x) \notag\\ 
    & \leq \textnormal{P}_{X}(\hat{f}_{n,a} \neq f^*) \sup_{x \in [0,1]^k} |\hat{f}_{n,a}(x) - f^*(x)|. 
```

The right-hand side term in \ref{lemma 3.5 proof}, is yielded as follows. we know that for `x \in [0, 1]^k` whenever the `f_{n,a} \neq f^*(x)`, the value of `|1 - 2\eta(x)|` is in `[0, 1]`. On the other hand, we know that the ` |1 - 2\eta(x)| \leq \sup_{x \in [0,1]^k} |\hat{f}_{n,a}(x) - f^*(x)|`. Thus, by multiplying `\sup_{x \in [0,1]^k} |\hat{f}_{n,a}(x) - f^*(x)|` and `\textnormal{P}_{X}(\hat{f}_{n,a} \neq f^*)`, we achieve the upper bound for `\mathcal{E}(\hat{f}_{n,a})`.
\end{proof} 

In the next chapter, we will conduct an empirical comparison of the performance of the realizable and noisy cases algorithm in detecting rare events.

