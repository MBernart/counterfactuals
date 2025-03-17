## DiCE (Diverse Counterfactual Explanations)

DiCE represents a versatile approach to generating counterfactual explanations with minimal assumptions about available data.
### Key Characteristics

- Can generate explanations without access to training data, using only basic feature metadata
- Works with pre-trained models, making it practical for real-world deployment scenarios
- Uses genetic algorithms for generating diverse counterfactual explanations
- Supports both categorical and continuous features through metadata definitions

## FACE (Feasible and Actionable Counterfactual Explanations)

FACE specifically addresses limitations in earlier counterfactual methods by emphasizing practical usability.
### Key Characteristics

- Focuses on generating explanations that are both feasible and actionable
- Respects the underlying data distribution, avoiding unrealistic suggestions
- Connects explanations via high-density paths to the original instance
- Model-agnostic design works with various prediction models
### Advantages Over Other Methods

- Unlike some methods, it handles discrete features and their restrictions in a principled manner
- Supports features that cannot change or can only change within specified ranges
- Accommodates user preferences on subjective distance measures
- Produces explanations that avoid "unachievable goals" with undesired consequences

## Wachter's Counterfactual Explanations

Wachter's approach represents one of the seminal works in counterfactual explanations, with a focus on providing legally compliant explanations.
### Key Characteristics

- Defines counterfactual explanations as the minimal set of changes needed to achieve a desired outcome
- Designed with legal compliance in mind, particularly for GDPR's "right to explanation"
- Uses distance metrics (often Manhattan distance) to find minimal changes
- Presents explanations in natural language format accessible to laypeople

## CADEX (Constrained Adversarial Examples)

CADEX focuses on the practical implementation of counterfactual explanations in business settings.
### Key Characteristics

- Incorporates business or domain constraints directly into the explanation generation process
- Handles categorical attributes and range constraints in a structured manner
- Provides actionable suggestions rather than just explaining existing decisions
- Designed for real-world applications like banking and finance

CADEX addresses the question of "how a different outcome can be achieved" rather than merely explaining why a particular classification was made, giving recipients actionable paths forward.

## CEM (Contrastive Explanation Method)

CEM approaches counterfactual explanation through a contrastive lens, focusing on what is both present and absent.
### Key Characteristics

- Generates explanations highlighting both pertinent positives (what should be present) and pertinent negatives (what should be absent)
- Optimizes for minimal changes while maintaining prediction confidence
- Addresses the human preference for contrastive explanations
## Growing Spheres

Growing Spheres offers a geometrically-inspired approach to finding counterfactual explanations.
### Key Characteristics

- Uses an expanding sphere in feature space to search for counterfactual instances
- Implements a binary search mechanism to find the closest valid counterfactual
- Balances computational efficiency with counterfactual proximity

## Actionable Recourse

Actionable Recourse specifically focuses on providing practical paths for users to change unfavorable outcomes.
### Key Characteristics

- Emphasizes feasibility and actionability of suggested changes
- Accounts for causal relationships between features
- Considers the cost of feature modifications in generating recommendations
- Aims to provide meaningful options for users affected by algorithmic decisions

## CFProto (Counterfactual Prototypes)

CFProto leverages the concept of prototypes to generate more intuitive counterfactual explanations.
### Key Characteristics

- Generates counterfactuals that are close to existing data prototypes
- Aims for more natural and intuitive explanations through prototype similarity
- Balances closeness to the original instance with closeness to class prototypes

## FIMAP (Feature Importance by Minimal Perturbation)

FIMAP explores feature importance through the lens of minimal perturbations.
### Key Characteristics

- Assigns importance to features based on how small perturbations affect predictions
- Combines feature importance with counterfactual generation
- Provides both global and local interpretability

# Comparative Analysis

## Approach to Feasibility and Actionability

The methods vary significantly in how they address the practical constraints of counterfactual explanations:

- **Strong emphasis**: FACE and Actionable Recourse explicitly focus on generating explanations that are both feasible and actionable
- **Moderate emphasis**: CADEX and CFProto incorporate business or domain constraints and prototype similarity
- **Limited emphasis**: Wachter's approach primarily focuses on finding minimal changes without explicitly addressing feasibility

## Data and Model Requirements

Different methods make different assumptions about data and model access:

- **Minimal requirements**: DiCE can operate with just model access and feature metadata, without training data
- **Model-agnostic approaches**: FACE, Wachter, and most others can work with any predictive model type[
- **Specific model types**: Some methods may have optimizations for particular model architectures

## Feature Type Handling

The ability to handle different types of features varies:

- **Comprehensive handling**: FACE and CADEX explicitly support categorical features, constraints, and user preferences
- **Basic handling**: DiCE supports categorical and continuous features through metadata definitions
- **Varied approaches**: Other methods have different mechanisms for handling feature types and constraints
