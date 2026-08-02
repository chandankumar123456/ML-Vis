// src/pages/JourneyPage.tsx
import { Link } from 'react-router-dom';
import { useProgressStore } from '../store/progressStore';

// Curated progression path — mirrors the syllabus's conceptual chain
const JOURNEY = [
  { id: 'simple-linear-regression', label: 'Linear Regression' },
  { id: 'gradient-descent', label: 'Gradient Descent' },
  { id: 'ridge-regression', label: 'Ridge Regression' },
  { id: 'lasso-regression', label: 'LASSO Regression' },
  { id: 'logistic-regression', label: 'Logistic Regression' },
  { id: 'softmax-regression', label: 'Softmax Regression' },
  { id: 'knn', label: 'K-NN' },
  { id: 'naive-bayes', label: 'Naive Bayes' },
  { id: 'svm-hard-margin', label: 'SVM' },
  { id: 'perceptron', label: 'Perceptron' },
  { id: 'decision-trees', label: 'Decision Trees' },
  { id: 'kmeans', label: 'K-Means' },
  { id: 'pca', label: 'PCA' },
  { id: 'neural-networks', label: 'Neural Networks' },
  { id: 'backpropagation', label: 'Backpropagation' },
];

export function JourneyPage() {
  const completed = useProgressStore((s) => s.completed);
  return (
    <div>
      <h1>Learning Journey</h1>
      <p>Follow the conceptual progression — each concept builds on the last.</p>
      <div className="journey">
        {JOURNEY.map((j, i) => {
          const done = !!completed[j.id];
          return (
            <div key={j.id} className="journey-node">
              <Link to={`/topic/${j.id}`} className={done ? 'journey-link done' : 'journey-link'}>
                <div className="journey-dot">{done ? '✓' : i + 1}</div>
                {j.label}
              </Link>
              {i < JOURNEY.length - 1 && <div className="journey-arrow">↓</div>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
