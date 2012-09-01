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

public class ProblemVariations implements AnnealAdapter<SimpleProblem> {
	
	@Override
	public Set<SimpleProblem> mutations(SimpleProblem p, final Random rand) {
		p = new SimpleProblem(p.x,p.y,p.wt); // copy so any alterations we make are not visible outside here (thread safety)
		final Set<SimpleProblem> mutations = new TreeSet<SimpleProblem>();
		final int m = p.nrow;
		for(int i=0;i<m;++i) {
			final int wi = p.wt[i];
			final boolean yi = p.y[i];
			// row deletion
			{
				p.wt[i] = 0;
				final SimpleProblem np = SimpleProblem.cleanRep(p.x,p.y,p.wt);
				if((null!=np)&&(np.nrow>1)) {
					mutations.add(np);
				}
			}
			// weight changes
			{
				p.wt[i] = wi + 1;
				final SimpleProblem np = SimpleProblem.cleanRep(p.x,p.y,p.wt);
				if((null!=np)&&(np.nrow>1)) {
					mutations.add(np);
				}
			}
			{
				p.wt[i] = wi - 1;
				final SimpleProblem np = SimpleProblem.cleanRep(p.x,p.y,p.wt);
				if((null!=np)&&(np.nrow>1)) {
					mutations.add(np);
				}
			}
			p.wt[i] = wi;
			// y-flip
			{
				p.y[i] = !yi;
				final SimpleProblem np = SimpleProblem.cleanRep(p.x,p.y,p.wt);
				if((null!=np)&&(np.nrow>1)) {
					mutations.add(np);
				}
			}
			p.y[i] = yi;
			// x-changes
			for(int j=0;j<p.dim;++j) {
				final double xij = p.x[i][j];
				{
					p.x[i][j] = xij + 1.0;
					final SimpleProblem np = SimpleProblem.cleanRep(p.x,p.y,p.wt);
					if((null!=np)&&(np.nrow>1)) {
						mutations.add(np);
					}
				}
				{
					p.x[i][j] = xij - 1.0;
					final SimpleProblem np = SimpleProblem.cleanRep(p.x,p.y,p.wt);
					if((null!=np)&&(np.nrow>1)) {
						mutations.add(np);
					}
				}
				{
					p.x[i][j] = -xij;
					final SimpleProblem np = SimpleProblem.cleanRep(p.x,p.y,p.wt);
					if((null!=np)&&(np.nrow>1)) {
						mutations.add(np);
					}
				}
				{
					p.x[i][j] = 1.1*xij;
					final SimpleProblem np = SimpleProblem.cleanRep(p.x,p.y,p.wt);
					if((null!=np)&&(np.nrow>1)) {
						mutations.add(np);
					}
				}
				{
					p.x[i][j] = 0.9*xij;
					final SimpleProblem np = SimpleProblem.cleanRep(p.x,p.y,p.wt);
					if((null!=np)&&(np.nrow>1)) {
						mutations.add(np);
					}
				}
				p.x[i][j] = xij;
			}
			// make sure all restored
			p.wt[i] = wi;
			p.y[i] = yi;
		}
		return mutations;
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
		final int[] w = new int[m];
		for(int rep=0;rep<10;++rep) {
			for(int i=0;i<m;++i) {
				w[i] = rand.nextBoolean()?ow[i]:0;
			}
			final SimpleProblem np = SimpleProblem.cleanRep(x,y,w);
			if((null!=np)&&(np.nrow>1)&&(np.nrow<=20)) {
				children.add(np);
			}
		}
		return children;
	}
	
	public static double scoreExample(final double[][] x,
			final boolean[] y, final int[] wt) {
		if(x.length<1) {
			return 0.0;
		}
		final int dim = x[0].length;
		// real crude approximate boundedness check
		final boolean[] sawAgreement = new boolean[dim];
		final boolean[] sawDisaggree = new boolean[dim];
		for(int i=0;i<x.length;++i) {
			for(int j=0;j<dim;++j) {
				if(Math.abs(x[i][j])>1.0e-8) {
					if(y[i]==(x[i][j]>0)) {
						sawAgreement[j] |= true;
					} else { 
						sawDisaggree[j] |= true;
					}
				}
			}
		}
		for(int j=0;j<dim;++j) {
			if(!sawAgreement[j]) {
				return 0.0;
			}
			if(!sawDisaggree[j]) {
				return 0.0;
			}
		}
		final DoubleMatrix1D wts = new DenseDoubleMatrix1D(dim);
		final double perplexity0 = perplexity(x,y,wt,wts);
		double perplexity1 = 0.0;
		for(int ns=0;ns<=10;++ns) {
			final boolean sawDiff = NewtonStep(x,y,wt,wts, false, ns<=0?1.0e-3:0.0);
			if(!sawDiff) {
				return 0.0;
			}
			final double perplexityI = perplexity(x,y,wt,wts);
			if(ns<=0) {
				perplexity1 = perplexityI;
			}
			if(perplexityI>perplexity0) {
				// don't count perplexity of things too near start (they can be rounding error)
				boolean sawNZ = false;
				for(int j=0;j<dim;++j) {
					if(Math.abs(wts.get(j))>1.0e-3) {
						sawNZ = true;
						break;
					}
				}
				if(sawNZ) {
					return perplexity1/(perplexity0*(1.0+ns));
				}
			}
		}
		return 0.0;
	}
	
	@Override
	public double scoreExample(final SimpleProblem p) {
		return scoreExample(p.x,p.y,p.wt);
	}

	/**
	 * @param x
	 * @param y
	 * @param wt data weights (all > 0)
	 * @param verbose print a lot
	 * @param minAbsDet TODO
	 * @param update control if step is taken and wts are updated (inefficient way to compute score, but useful for debugging)
	 * @return true if steped
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
