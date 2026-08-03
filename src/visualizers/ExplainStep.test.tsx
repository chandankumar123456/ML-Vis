// Component-level coverage for the explain-step registry view (Task 5):
// collapsed-by-default toggle, expanded changed/why/dependsOn/gateConcepts
// sections, formulaRef resolution against topic.formulas, safe no-snapshot
// state, and aria-expanded toggling.
import { describe, it, expect } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import { ExplainStep } from './ExplainStep';
import { gdModule } from '../topics/gradient-descent/module';
import type { SimState, StepExplanation } from '../engine/types';

const mkSnapshot = (explanation: StepExplanation): SimState => ({
  algorithm: {},
  visuals: [],
  math: [],
  narration: '',
  explanation,
  highlights: [],
  metrics: {},
  events: [],
  timeline: [],
});

const baseExplanation: StepExplanation = {
  changed: ['θ: 2.30 → 2.12', 'x₀'],
  why: 'gradient was positive, so the weight moves opposite the gradient.',
  dependsOn: ['gradient', 'learning rate'],
  gateConcepts: ['Optimization', 'Calculus'],
};

describe('ExplainStep', () => {
  it('renders a collapsed panel with the toggle when a snapshot is provided', () => {
    render(<ExplainStep snapshot={mkSnapshot(baseExplanation)} />);
    const toggle = screen.getByRole('button', { name: /Why did this step change/i });
    expect(toggle).toBeInTheDocument();
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
    // expanded content hidden while collapsed
    expect(screen.queryByText(/gradient was positive/i)).toBeNull();
  });

  it('expanding shows changed chips, why prose, dependsOn and gateConcepts', () => {
    render(<ExplainStep snapshot={mkSnapshot(baseExplanation)} />);
    fireEvent.click(screen.getByRole('button', { name: /Why did this step change/i }));

    expect(screen.getByText('θ: 2.30 → 2.12')).toBeInTheDocument();
    expect(screen.getByText('x₀')).toBeInTheDocument();
    expect(screen.getByText(/gradient was positive/i)).toBeInTheDocument();
    expect(screen.getByText('gradient')).toBeInTheDocument();
    expect(screen.getByText('learning rate')).toBeInTheDocument();
    expect(screen.getByText('Optimization')).toBeInTheDocument();
    expect(screen.getByText('Calculus')).toBeInTheDocument();
  });

  it('resolves formulaRef against the topic formulas when found', () => {
    render(
      <ExplainStep snapshot={mkSnapshot({ ...baseExplanation, formulaRef: 'grad' })}
        topic={gdModule} />
    );
    fireEvent.click(screen.getByRole('button', { name: /Why did this step change/i }));

    expect(screen.getByText('grad')).toBeInTheDocument();
    expect(screen.getByText('\\frac{df}{dx} = 2x')).toBeInTheDocument();
  });

  it('renders the bare id when formulaRef is unknown', () => {
    render(
      <ExplainStep snapshot={mkSnapshot({ ...baseExplanation, formulaRef: 'missing-id' })}
        topic={gdModule} />
    );
    fireEvent.click(screen.getByRole('button', { name: /Why did this step change/i }));

    expect(screen.getByText('missing-id')).toBeInTheDocument();
    expect(screen.queryByText(/\\frac\{df\}\{dx\}/)).toBeNull();
  });

  it('renders nothing when there is no snapshot', () => {
    const { container } = render(<ExplainStep />);
    expect(container).toBeEmptyDOMElement();
  });

  it('toggles aria-expanded as the panel opens and closes', () => {
    render(<ExplainStep snapshot={mkSnapshot(baseExplanation)} />);
    const toggle = screen.getByRole('button', { name: /Why did this step change/i });
    expect(toggle).toHaveAttribute('aria-expanded', 'false');

    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'true');

    fireEvent.click(toggle);
    expect(toggle).toHaveAttribute('aria-expanded', 'false');
  });
});
