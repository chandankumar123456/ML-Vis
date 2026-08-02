// src/visualizers/knowledgeGraph/graphData.ts
export type EdgeType =
  | 'requires' | 'related' | 'extends' | 'derives-from'
  | 'contrasts-with' | 'frequently-confused' | 'hidden-gate-link';

export interface GraphNode {
  id: string;
  label: string;
  kind: 'topic' | 'concept' | 'module';
  category: 'regression' | 'classification' | 'dim-reduction' | 'trees' | 'clustering' | 'nn' | 'evaluation' | 'foundation';
  weight: number;   // GATE weightage 1-5
}

export interface GraphEdge {
  source: string;
  target: string;
  type: EdgeType;
  note: string;     // "why this connection"
}

export const graphNodes: GraphNode[] = [
  { id: 'linear-algebra', label: 'Linear Algebra', kind: 'module', category: 'foundation', weight: 5 },
  { id: 'calculus', label: 'Calculus', kind: 'module', category: 'foundation', weight: 5 },
  { id: 'probability', label: 'Probability', kind: 'module', category: 'foundation', weight: 4 },
  { id: 'projection', label: 'Projection', kind: 'concept', category: 'foundation', weight: 3 },
  { id: 'eigenvalue', label: 'Eigenvalues', kind: 'concept', category: 'foundation', weight: 4 },
  { id: 'simple-linear-regression', label: 'Linear Regression', kind: 'topic', category: 'regression', weight: 5 },
  { id: 'gradient-descent', label: 'Gradient Descent', kind: 'topic', category: 'foundation', weight: 5 },
  { id: 'regularization', label: 'Regularization', kind: 'concept', category: 'regression', weight: 4 },
  { id: 'ridge-regression', label: 'Ridge Regression', kind: 'topic', category: 'regression', weight: 4 },
  { id: 'lasso-regression', label: 'LASSO Regression', kind: 'topic', category: 'regression', weight: 4 },
  { id: 'logistic-regression', label: 'Logistic Regression', kind: 'topic', category: 'classification', weight: 5 },
  { id: 'cross-entropy', label: 'Cross Entropy', kind: 'concept', category: 'classification', weight: 4 },
  { id: 'mle', label: 'MLE', kind: 'concept', category: 'classification', weight: 3 },
  { id: 'softmax-regression', label: 'Softmax Regression', kind: 'topic', category: 'classification', weight: 4 },
  { id: 'neural-networks', label: 'Neural Networks', kind: 'topic', category: 'nn', weight: 5 },
  { id: 'backpropagation', label: 'Backpropagation', kind: 'topic', category: 'nn', weight: 5 },
  { id: 'pca', label: 'PCA', kind: 'topic', category: 'dim-reduction', weight: 5 },
  { id: 'pca-svd', label: 'PCA & SVD', kind: 'topic', category: 'dim-reduction', weight: 4 },
  { id: 'svm-hard-margin', label: 'SVM (Hard)', kind: 'topic', category: 'classification', weight: 5 },
  { id: 'svm-soft-margin', label: 'SVM (Soft)', kind: 'topic', category: 'classification', weight: 5 },
  { id: 'perceptron', label: 'Perceptron', kind: 'topic', category: 'classification', weight: 3 },
  { id: 'lda', label: 'LDA', kind: 'topic', category: 'dim-reduction', weight: 3 },
  { id: 'knn', label: 'K-NN', kind: 'topic', category: 'classification', weight: 3 },
  { id: 'naive-bayes', label: 'Naive Bayes', kind: 'topic', category: 'classification', weight: 3 },
  { id: 'decision-trees', label: 'Decision Trees', kind: 'topic', category: 'trees', weight: 4 },
  { id: 'kmeans', label: 'K-Means', kind: 'topic', category: 'clustering', weight: 3 },
  { id: 'hierarchical-clustering', label: 'Hierarchical Clustering', kind: 'topic', category: 'clustering', weight: 2 },
  { id: 'bias-variance', label: 'Bias-Variance', kind: 'topic', category: 'evaluation', weight: 4 },
  { id: 'cross-validation', label: 'Cross Validation', kind: 'topic', category: 'evaluation', weight: 4 },
  { id: 'classification-metrics', label: 'Precision/Recall/F1', kind: 'topic', category: 'evaluation', weight: 4 },
  { id: 'roc-auc', label: 'ROC/AUC', kind: 'topic', category: 'evaluation', weight: 4 },
  { id: 'sigmoid', label: 'Sigmoid', kind: 'concept', category: 'classification', weight: 4 },
  { id: 'entropy', label: 'Entropy', kind: 'concept', category: 'trees', weight: 4 },
  { id: 'margin', label: 'Margin', kind: 'concept', category: 'classification', weight: 4 },
];

export const graphEdges: GraphEdge[] = [
  // foundational chains (hidden-gate-links from other modules)
  { source: 'linear-algebra', target: 'projection', type: 'requires', note: 'Projection is a linear algebra operation' },
  { source: 'projection', target: 'simple-linear-regression', type: 'hidden-gate-link', note: 'OLS = projection onto column space' },
  { source: 'calculus', target: 'gradient-descent', type: 'requires', note: 'GD needs partial derivatives' },
  { source: 'probability', target: 'mle', type: 'requires', note: 'MLE is a probabilistic framework' },
  { source: 'eigenvalue', target: 'pca', type: 'hidden-gate-link', note: 'PCA solves an eigenproblem' },
  // ML chain
  { source: 'simple-linear-regression', target: 'gradient-descent', type: 'requires', note: 'GD minimizes OLS cost' },
  { source: 'simple-linear-regression', target: 'ridge-regression', type: 'extends', note: 'Ridge = LR + L2 penalty' },
  { source: 'ridge-regression', target: 'lasso-regression', type: 'contrasts-with', note: 'L1 vs L2 geometry: diamond vs circle' },
  { source: 'regularization', target: 'ridge-regression', type: 'derives-from', note: 'Ridge is the canonical L2 regularizer' },
  { source: 'gradient-descent', target: 'logistic-regression', type: 'requires', note: 'Logistic trained by GD' },
  { source: 'mle', target: 'cross-entropy', type: 'derives-from', note: 'CE = negative log likelihood' },
  { source: 'cross-entropy', target: 'logistic-regression', type: 'derives-from', note: 'Logistic uses CE loss' },
  { source: 'logistic-regression', target: 'softmax-regression', type: 'extends', note: 'Softmax generalizes sigmoid to K classes' },
  { source: 'sigmoid', target: 'logistic-regression', type: 'requires', note: 'Sigmoid maps logits to probabilities' },
  { source: 'cross-entropy', target: 'neural-networks', type: 'requires', note: 'NNs minimize CE' },
  { source: 'logistic-regression', target: 'neural-networks', type: 'requires', note: 'Logistic unit = single neuron' },
  { source: 'neural-networks', target: 'backpropagation', type: 'requires', note: 'Backprop computes NN gradients' },
  { source: 'gradient-descent', target: 'backpropagation', type: 'requires', note: 'Backprop feeds GD updates' },
  { source: 'pca', target: 'pca-svd', type: 'extends', note: 'PCA computed via SVD' },
  { source: 'pca', target: 'lda', type: 'frequently-confused', note: 'Both reduce dims; LDA uses labels' },
  { source: 'svm-hard-margin', target: 'svm-soft-margin', type: 'extends', note: 'Soft margin allows slack' },
  { source: 'svm-hard-margin', target: 'perceptron', type: 'frequently-confused', note: 'Both find separating hyperplanes' },
  { source: 'margin', target: 'svm-hard-margin', type: 'requires', note: 'SVM maximizes margin' },
  { source: 'knn', target: 'naive-bayes', type: 'frequently-confused', note: 'Both are instance/probabilistic classifiers' },
  { source: 'entropy', target: 'decision-trees', type: 'requires', note: 'Trees split by entropy/IG' },
  { source: 'bias-variance', target: 'classification-metrics', type: 'related', note: 'Metrics measure the tradeoff' },
  { source: 'classification-metrics', target: 'roc-auc', type: 'extends', note: 'ROC uses TPR/FPR from confusion matrix' },
  { source: 'perceptron', target: 'logistic-regression', type: 'related', note: 'Same linear form, different loss' },
  { source: 'gradient-descent', target: 'perceptron', type: 'requires', note: 'Perceptron rule ≈ SGD variant' },
];

export function edgesOf(nodeId: string): GraphEdge[] {
  return graphEdges.filter((e) => e.source === nodeId || e.target === nodeId);
}

export function edgeTypeColor(t: EdgeType): string {
  switch (t) {
    case 'requires': return '#3b82f6';
    case 'related': return '#8b5cf6';
    case 'extends': return '#10b981';
    case 'derives-from': return '#f59e0b';
    case 'contrasts-with': return '#ef4444';
    case 'frequently-confused': return '#ec4899';
    case 'hidden-gate-link': return '#64748b';
  }
}

export function nodeColor(cat: string): string {
  switch (cat) {
    case 'regression': return '#3b82f6';
    case 'classification': return '#8b5cf6';
    case 'dim-reduction': return '#10b981';
    case 'trees': return '#f59e0b';
    case 'clustering': return '#ef4444';
    case 'nn': return '#ec4899';
    case 'evaluation': return '#14b8a6';
    default: return '#64748b';
  }
}
