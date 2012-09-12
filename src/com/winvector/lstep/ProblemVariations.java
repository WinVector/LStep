package com.winvector.lstep;

import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;

import com.winvector.anneal.AnnealAdapter;

public final class ProblemVariations implements AnnealAdapter<SimpleProblem> {
	
	private static void collectProblem(final Random rand, final Set<SimpleProblem> c, final double[][] x, final boolean[] y, final int[] w) {
		if(acceptableProblem(x,y,w)) {
			final SimpleProblem np = SimpleProblem.cleanRep(x,y,w);
			if(null!=np) {
				fixQtoOneHalf(rand,np.y,np.wt);
				if(acceptableProblem(np.x,np.y,np.wt)) {
					c.add(np);
				}
			}
		}
	}
	
	@Override
	public Set<SimpleProblem> mutations(SimpleProblem p, final Random rand) {
		p = new SimpleProblem(p.x,p.y,p.wt); // copy so any alterations we make are not visible outside here (thread safety)
		final Set<SimpleProblem> mutations = new TreeSet<SimpleProblem>();
		final int m = p.nrow;
		for(int i=0;i<m;++i) {
			final int wi = p.wt[i];
			final boolean yi = p.y[i];
			// weight changes
			for(int v=0;v<=100;++v) {
				p.wt[i] = v;
				collectProblem(rand,mutations,p.x,p.y,p.wt);
			}
			for(int j=0;j<m;++j) {
				if(i!=j) {
					final int wj = p.wt[j];
					p.wt[i] = wi + 1;
					p.wt[j] = wj - 1;
					collectProblem(rand,mutations,p.x,p.y,p.wt);
					p.wt[i] = wi;
					p.wt[j] = wj;
				}
			}
			// y-flip
			{
				p.y[i] = !yi;
				collectProblem(rand,mutations,p.x,p.y,p.wt);
			}
			p.y[i] = yi;
			// x-changes
			for(int j=1;j<p.dim;++j) {
				final double xij = p.x[i][j];
				{
					p.x[i][j] = xij + 1.0;
					collectProblem(rand,mutations,p.x,p.y,p.wt);
				}
				{
					p.x[i][j] = xij - 1.0;
					collectProblem(rand,mutations,p.x,p.y,p.wt);
				}
				{
					p.x[i][j] = -xij;
					collectProblem(rand,mutations,p.x,p.y,p.wt);
				}
				{
					p.x[i][j] = 1.1*xij;
					collectProblem(rand,mutations,p.x,p.y,p.wt);
				}
				{
					p.x[i][j] = 0.9*xij;
					collectProblem(rand,mutations,p.x,p.y,p.wt);
				}
				for(int k=0;k<10;++k) {
					p.x[i][j] += rand.nextGaussian();
					collectProblem(rand,mutations,p.x,p.y,p.wt);
				}
				// Gibbs like line sweep
				for(int v=-100;v<=100;v+=1) {
					p.x[i][j] = v;
					collectProblem(rand,mutations,p.x,p.y,p.wt);
				}
				p.x[i][j] = xij;
			}
			// make sure all restored
			p.wt[i] = wi;
			p.y[i] = yi;
		}
		// subsets
		addSubsets(mutations,p.x,p.y,p.wt,rand);
		// scramble y's (destructive to p)
		for(int s=0;s<100;++s) {
			for(int i=0;i<m;++i) {
				p.y[i] = rand.nextBoolean();
				collectProblem(rand,mutations,p.x,p.y,p.wt);
			}
		}
		return mutations;
	}
	
	private static void addSubsets(final Set<SimpleProblem> c, final double[][] x, final boolean[] y, final int[] ow, final Random rand) {
		final int m = x.length;
		final int[] w = new int[m];
		for(int rep=0;rep<100;++rep) {
			for(int i=0;i<m;++i) {
				w[i] = rand.nextBoolean()?ow[i]:0;
			}
			collectProblem(rand,c,x,y,w);
		}		
	}
	
	@Override
	public Set<SimpleProblem> breed(final SimpleProblem p0, final SimpleProblem p1, final Random rand) {			
		final Set<SimpleProblem> children = new TreeSet<SimpleProblem>();
		final int dim = p0.dim;
		final int m = p0.nrow + p1.nrow;
		final double[][] x = new double[m][dim];
		final boolean[] y = new boolean[m];
		final int[] ow = new int[m];
		for(int i=0;i<p0.nrow;++i) {
			for(int j=0;j<dim;++j) {
				x[i][j] = p0.x[i][j];
			}
			y[i] = p0.y[i];
			ow[i] = p0.wt[i];
		}
		for(int i=0;i<p1.nrow;++i) {
			for(int j=0;j<dim;++j) {
				x[p0.nrow+i][j] = p1.x[i][j];
			}
			y[p0.nrow+i] = p1.y[i];
			ow[p0.nrow+i] = p1.wt[i];
		}
		addSubsets(children,x,y,ow,rand);
		return children;
	}
	
	public static boolean acceptableProblem(final double[][] x,
			final boolean[] y, final int[] wt) {
		if((null==x)||(x.length<=1)||(x[0].length<1)) {
			return false;
		}
		final int dim = x[0].length;
		// real crude approximate boundedness check
		final boolean[] sawAgreement = new boolean[dim];
		final boolean[] sawDisaggree = new boolean[dim];
		int nNZRow = 0;
		for(int i=0;i<x.length;++i) {
			final int wti = (null==wt)?1:wt[i];
			if(wti>0) {
				if(wti>100) {
					return false;
				}
				++nNZRow;
				for(int j=0;j<dim;++j) {
					if(Math.abs(x[i][j])>100.0) {
						return false;
					}
					if(Math.abs(x[i][j])>1.0e-8) {
						if(y[i]==(x[i][j]>0)) {
							sawAgreement[j] |= true;
						} else { 
							sawDisaggree[j] |= true;
						}
					}
				}
			}
		}
		if(nNZRow<=1) {
			return false;
		}
		for(int j=0;j<dim;++j) {
			if(!sawAgreement[j]) {
				return false;
			}
			if(!sawDisaggree[j]) {
				return false;
			}
		}
		return true;
	}
	
	
	public static double scoreExample(final double[][] x,
			final boolean[] y, final int[] wt) {
		if(!acceptableProblem(x,y,wt)) {
			return 0.0;
		}
		final int dim = x[0].length;
		final DoubleMatrix1D wts = new DenseDoubleMatrix1D(dim);
		final double perplexity0 = perplexity(x,y,wt,wts);
		final boolean sawDiff = NewtonStep(x,y,wt,wts, false, 1.0e-3);
		if(!sawDiff) {
			return 0.0;
		}
		double delta0 = 0.0;
		for(int j=0;j<dim;++j) {
			delta0 += wts.get(j)*wts.get(j);
		}
		if(delta0<=1.0e-8) {
			return 0.0;
		}
		final double perplexity1 = perplexity(x,y,wt,wts);
		if(perplexity1>perplexity0) {
			return (perplexity1/perplexity0)*1000.0;  
		} else {
			return 1.0/(1.01-perplexity1/perplexity0);
		}
	}
	
	@Override
	public double scoreExample(final SimpleProblem p) {
		return scoreExample(p.x,p.y,p.wt);
	}

	/**
	 * assumes there are plus and minus examples with non-zero weights (implied by acceptable problem is true)
	 * @param rand
	 * @param y
	 * @param w
	 */
	public static void fixQtoOneHalf(final Random rand, final boolean[] y, final int[] w) { 
		// get weights to 1/2 without changing zeronoess/non-zeroness of weights
		final int m = y.length;
		long wPlus = 0;
		long wMinus = 0;
		for(int i=0;i<m;++i) {
			if(y[i]) {
				wPlus += w[i];
			} else {
				wMinus += w[i];
			}
		}
		while(wPlus!=wMinus) {
			final int i = rand.nextInt(m);
			if(w[i]>0) {
				final int delta = y[i]==(wMinus>wPlus)?1:-1;
				if(w[i]+delta>0) {
					w[i] += delta;
					if(y[i]) {
						wPlus += delta;
					} else {
						wMinus += delta;
					}
				}
			}
		}
	}

	/**
	 * @param x
	 * @param y
	 * @param wt data weights (all > 0)
	 * @param verbose print a lot
	 * @param minAbsDet TODO
	 * @param update control if step is taken and wts are updated (inefficient way to compute score, but useful for debugging)
	 * @return true if stepped
	 */
	public static final boolean NewtonStep(final double[][] x, final boolean[] y, final int[] wt, final DoubleMatrix1D wts, final boolean verbose, double minAbsDet) {
		final int nDat = x.length;
		final int dim = x[0].length;
		final DoubleMatrix2D m = new DenseDoubleMatrix2D(dim,dim);
		final DoubleMatrix2D v = new DenseDoubleMatrix2D(dim,1);
		for(int i=0;i<nDat;++i) {
			if((null==wt)||(wt[i]>0)) {
				final double wti = wt==null?1.0:wt[i];
				final double[] xi = x[i];
				final double pi = ScoreStep.sigmoid(ScoreStep.dot(wts,xi));
				final double mwt = pi*(1.0-pi);
				final double vwt = (y[i]?1.0:0.0) - pi;
				for(int j=0;j<dim;++j) {
					v.set(j,0,v.get(j,0) + wti*vwt*xi[j]);
					for(int k=0;k<dim;++k) {
						m.set(j,k,m.get(j,k) + wti*mwt*xi[j]*xi[k]);
					}
				}
			}
		}
		try {
			if(verbose) {
				System.out.println("m:\n" + m);
				System.out.println("v:\n" + v);
			}
			if(minAbsDet>0) {
				final double det = Algebra.DEFAULT.det(m);
				if(Math.abs(det)<minAbsDet) {
					return false;
				}
			}
			final DoubleMatrix2D delta = Algebra.DEFAULT.solve(m, v);
			if(verbose) {
				System.out.println("delta:\n" + delta);
			}
			boolean sawDiff = false;
			for(int j=0;j<dim;++j) {
				final double wi = wts.get(j);
				final double nwi = wi+delta.get(j,0);
				if(Math.abs(wi-nwi)/(Math.max(1.0,Math.abs(wi)))>1.0e-6) {
					sawDiff = true;
				}
				wts.set(j,nwi);
			}
			return sawDiff;
		} catch (Exception ex) {
		}
		return false;
	}

	public static final double perplexity(final double[][] x, final boolean[] y, final int[] wt, final DoubleMatrix1D wts) {
		final int nDat = x.length;
		double perplexity = 0.0;
		for(int i=0;i<nDat;++i) {
			if((null==wt)||(wt[i]>0)) {
				final double wti = wt==null?1.0:wt[i];
				final double[] xi = x[i];
				final double d = ScoreStep.dot(wts,xi);				
				if(y[i]) {
					perplexity -= wti*ScoreStep.logSigmoid(d);
				} else {
					perplexity -= wti*ScoreStep.logOneMinusSigmoid(d);
				}
			}
		}
		return perplexity;
	}
}
