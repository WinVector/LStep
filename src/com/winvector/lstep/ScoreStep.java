package com.winvector.lstep;

import java.util.Date;
import java.util.Random;
import java.util.Set;
import java.util.TreeSet;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;


/**
 * quick demonstration of Newton-Raphson divergence on logistic regression based on a quick implementation of
 * http://www.win-vector.com/blog/2011/09/the-simpler-derivation-of-logistic-regression/
 * 
 * used as basis for articles:
 *   http://www.win-vector.com/blog/2012/08/how-robust-is-logistic-regression/
 *   http://www.win-vector.com/blog/2012/08/newton-raphson-can-compute-an-average/
 * @author johnmount
 *
 */
public final class ScoreStep {

	private static final double ln0p5 = Math.log(0.5);

	private static double dot(final DoubleMatrix1D a, final double[] b) {
		final int n = b.length;
		double r = 0.0;
		for(int j=0;j<n;++j) {
			r += a.get(j)*b[j];
		}
		return r;
	}
	
	static double simpleSigmoid(final double x) {
		return 1.0/(1.0 + Math.exp(-x));
	}
	
	
	public static double sigmoid(final double x) {
		if(x>0) {
			return 1.0/(1.0 + Math.exp(-x));
		} else if(x<0) {
			final double v = Math.exp(x);
			return v/(1.0+v);
		} else {
			return 0.5;
		}
	}

	private static final double switchPt = 20.0;
	private static final double fSwitch = Math.log1p(Math.exp(switchPt));
	private static final double dSwitch = sigmoid(switchPt);
	public static double log1pExp(final double x) {
		if(x>=switchPt) {
			return fSwitch + dSwitch*(x-switchPt); //2 term Taylor series from switch point
		}
		return Math.log1p(Math.exp(x));
	}

	public static double logSigmoid(final double x) {
		if(x>0) {
			return -log1pExp(-x);
		} else if(x<0) {
			return x - log1pExp(x);
		} else {
			return ln0p5;
		}
	}
	
	public static double logOneMinusSigmoid(final double x) {
		if(x>0) {
			return -x - log1pExp(-x);
		} else if(x<0) {
			return -log1pExp(x);
		} else {
			return ln0p5;
		}
	}
	
	public static final double perplexity(final double[][] x, final boolean[] y, final int[] wt, final DoubleMatrix1D wts) {
		final int nDat = x.length;
		double perplexity = 0.0;
		for(int i=0;i<nDat;++i) {
			if((null==wt)||(wt[i]>0)) {
				final double wti = wt==null?1.0:wt[i];
				final double[] xi = x[i];
				final double d = dot(wts,xi);				
				if(y[i]) {
					perplexity -= wti*logSigmoid(d);
				} else {
					perplexity -= wti*logOneMinusSigmoid(d);
				}
			}
		}
		return perplexity;
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
				final double pi = sigmoid(dot(wts,xi));
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
	

	// crummy model- but best soln at wts = 0
	// acceptance region all inside -6<=|x|,|y|<=5
	/**
	 * example that gets R 2.15.0 to fail: 
	 * > p <- data.frame(x=c(1,0,1,0),y=c(T,T,F,F))
	 * > summary(glm(y~x,data=p,family=binomial(link='logit'),start=c(-4,6)))
	 * 
	 *      * Call:
	 * glm(formula = y ~ x, family = binomial(link = "logit"), data = p, 
	 *     start = c(-4, 6))
	 * 
	 * perplexity Residuals: 
	 *       1        2        3        4  
	 *  0.9737   0.0000  -1.3958  -8.4904  
	 * 
	 * Coefficients:
	 *               Estimate Std. Error   z value Pr(>|z|)    
	 * (Intercept)  2.252e+15  4.745e+07  47452996   <2e-16 ***
	 * x           -2.252e+15  4.745e+07 -47452996   <2e-16 ***
	 * ---
	 * Signif. codes:  0 Ô***Õ 0.001 Ô**Õ 0.01 Ô*Õ 0.05 Ô.Õ 0.1 Ô Õ 1 
	 * 
	 * (Dispersion parameter for binomial family taken to be 1)
	 * 
	 *     Null perplexity:  5.5452  on 3  degrees of freedom
	 * Residual perplexity: 74.9836  on 2  degrees of freedom
	 * AIC: 78.984
	 * 
	 * Number of Fisher Scoring iterations: 25
	 * 
	 * 
	 * 
	 * See also: http://andrewgelman.com/2011/05/whassup_with_gl/
	 * 
	 * articles pointing just to sep: 
	 *   http://www2.sas.com/proceedings/forum2008/360-2008.pdf
	 *   http://interstat.statjournals.net/YEAR/2011/articles/1110003.pdf
	 *   http://www.ats.ucla.edu/stat/mult_pkg/faq/general/complete_separation_logit_models.htm
	 * 
	 * see also:
	 * summary(glm(y~x,data=p,family=binomial(link='logit'),start=c(-4,6),maxit=1))
	 * summary(glm(y~x,data=p,family=binomial(link='logit'),start=c(-4,6),maxit=2))
	 */
	static SimpleProblem zeroSolnProblem = new SimpleProblem(
			new double[][] { 
					{ 1, 1 },
					{ 1, 0 },
					{ 1, 1 },
					{ 1, 0 }
			},
			new boolean[] {
					true,
					true,
					false,
					false
			},
			null
			);
	
	
	/**
	 * diverges from a zero start
	 * > d <- read.table('perp.tsv',sep='\t',header=T)
> ggplot(d,aes(x=wC,y=wX,z=perplexity0,fill=perplexity0)) + geom_tile(alpha=0.5) + scale_fill_gradient(low="green", high="red") + stat_contour()
	 * soln:  -4.603       -5.296  
	 * converges from: -4, -5 start
	 * ggplot(d,aes(x=wC,y=wX,z=increase,fill=increase)) + geom_tile(alpha=0.5)
	 */
	private static SimpleProblem badZeroStartProblem = new SimpleProblem(
			new double[][] { 
					{ 1, 0},
					{ 1, 0},
					{ 1, 0.001},
					{ 1, 100},
					{ 1, -1},
					{ 1, -1},
			},
			new boolean[] {
					false,
					true,
					false,
					false,
					false,
					true
			},
			new int[] {
					50,
					1,
					50,
					1,
					5,
					10
			}
			);
	
	public static void showProblem(final SimpleProblem prob, final double xL, final double xH, final double yL, final double yH) {
		final int dim = prob.dim;
		final int nPts = 100;
		final String sep = "\t";
		System.out.println("" + "wC" + sep + "wX" + sep + "perplexity0" + sep + "perplexity1" + sep + "decrease" + sep + "increase");
		for(int xi=0;xi<=nPts;++xi) {
			final double w0 = xL + (xH-xL)*xi/((double)nPts);
			for(int yi=0;yi<=nPts;++yi) {
				final double w1 = yL + (yH-yL)*yi/((double)nPts);
				final DoubleMatrix1D wts = new DenseDoubleMatrix1D(dim);
				wts.set(0,w0);
				wts.set(1,w1);
				final double perplexity0 = perplexity(prob.x,prob.y,prob.wt,wts);
				NewtonStep(prob.x,prob.y,prob.wt,wts, false, 0.0);
				final double perplexity1 = perplexity(prob.x,prob.y,prob.wt,wts);
				final boolean decrease = perplexity1<perplexity0;
				final boolean increase = perplexity1>perplexity0;
				System.out.println("" + w0 + sep + w1 + sep + perplexity0 + sep + perplexity1 + sep + decrease + sep + increase);
			}
		}
		/** 
		 * R-plot
		 * > d <- read.table('perp.tsv',sep='\t',header=T)
		 * > ggplot(d,aes(x=wC,y=wX,z=perplexity0,fill=perplexity0)) + geom_tile(alpha=0.5,binwidth=1) + scale_fill_gradient(low="green", high="red") + stat_contour(binwidth=1)
		 * > ggplot(d,aes(x=wC,y=wX,z=increase,fill=increase)) + geom_tile(alpha=0.5)
		 */
	}
	
	

	private static double scoreExample(final double[][] x,
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

	private static double scoreExample2(final double[][] x,
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
		final boolean sawDiff = NewtonStep(x,y,wt,wts, false, 1.0e-3);
		if(!sawDiff) {
			return 0.0;
		}
		// don't count perplexity of things too near start (they can be rounding error)
		boolean sawNZ = false;
		for(int j=0;j<dim;++j) {
			if(Math.abs(wts.get(j))>1.0e-3) {
				sawNZ = true;
				break;
			}
		}
		if(!sawNZ) {
			return 0.0;
		}
		final double perplexity1 = perplexity(x,y,wt,wts);
		return perplexity1/perplexity0;
	}

	public static Set<SimpleProblem> randProbs(final int dim) {
		final Set<SimpleProblem> found = new TreeSet<SimpleProblem>();
		final Random rand = new Random(3253);
		final int m = 20;
		final double[][] x = new double[m][dim];
		final boolean[] y = new boolean[m];
		final int[] w = new int[m];
		for(int rep=0;rep<10000;++rep) {
			for(int i=0;i<m;++i) {
				//x[i][0] = 1.0;
				for(int j=0;j<dim;++j) {
					x[i][j] = rand.nextGaussian();
				}
				y[i] = rand.nextBoolean();
				w[i] = rand.nextInt(100) - 50;
			}
			final SimpleProblem cleanRep = SimpleProblem.cleanRep(x,y,w);
			if((null!=cleanRep)&&(cleanRep.nrow>1)) {
				found.add(cleanRep);
			}			
		}
		return found;
	}
	
	public static Set<SimpleProblem> searchForProblem() {
		final Set<SimpleProblem> found = new TreeSet<SimpleProblem>();
		final double[][] x = { 
				{ 1, 0},
				{ 1, 0},
				{ 1, .001},
				{ 1, .001},
				{ 1, 1},
				{ 1, 1},
				{ 1, 100},
				{ 1, 100},
				{ 1, -.001},
				{ 1, -.001},
				{ 1, -1},
				{ 1, -1},
				{ 1, -100},
				{ 1, -100},
		};
		final int ndat = x.length;
		final boolean[] y = new boolean[ndat];
		for(int i=1;i<ndat;i+=2) {
			y[i] = true;
		}
		final Random rand = new Random(32535);
		final int[] wt = new int[ndat];
		for(int trial=0;trial<10000000;++trial) {
			for(int j=0;j<ndat;++j) {
				wt[j] = rand.nextInt(100) - 50;
			}
			final double score = scoreExample(x, y, wt);
			if(score>0.0) {
				final SimpleProblem cleanRep = SimpleProblem.cleanRep(x,y,wt);
				if((null!=cleanRep)&&(cleanRep.nrow>1)) {
					found.add(cleanRep);
					System.out.println("found: " + score + "\t" + cleanRep);
				}
			}
		}
		return found;
	}


	
	static final SimpleProblem example2D = new SimpleProblem(
			new double[][] { 
					{ 1.0,	0.0},
					{ 1.0,	0.0},
					{1.0,	0.001},
					{1.0,	100.0},
					{1.0,	-1.0},
					{1.0,	-1.0}
			},
			new boolean[] {
					false,
					true,
					false,
					false,
					false,
					true
			},
			new int[] {
					50,
					1,
					50,
					1,
					5,
					10
			}
			);
	/**
	 * R version:
	 * 	 * > d <- data.frame(x=c(0,0,0.001,100,-1,-1),y=c(F,T,F,F,F,T),wt=c(50,1,50,1,5,10))
	 * > glm(y~x,data=d,family=binomial(link='logit'),weights=d$wt)
	 * 
	 * Call:  glm(formula = y ~ x, family = binomial(link = "logit"), data = d, 
	 *     weights = d$wt)
	 * 
	 * Coefficients:
	 * (Intercept)            x  
	 *  -1.084e+15   -4.253e+13  
	 * 
	 * Degrees of Freedom: 5 Total (i.e. Null);  4 Residual
	 * Null Deviance:	    72.95 
	 * Residual Deviance: 793 	AIC: 797 
	 * Warning message:
	 * glm.fit: fitted probabilities numerically 0 or 1 occurred 
	 * > glm(y~x,data=d,family=binomial(link='logit'),weights=d$wt,start=c(-4,-5))
	 * 
	 * Call:  glm(formula = y ~ x, family = binomial(link = "logit"), data = d, 
	 *     weights = d$wt, start = c(-4, -5))
	 * 
	 * Coefficients:
	 * (Intercept)            x  
	 *      -4.603       -5.296  
	 * 
	 * Degrees of Freedom: 5 Total (i.e. Null);  4 Residual
	 * Null Deviance:	    72.95 
	 * Residual Deviance: 30.31 	AIC: 34.31 
	 * Warning message:
	 * glm.fit: fitted probabilities numerically 0 or 1 occurred 
	 * > 
	 * 
	 */
	
	public static void workProblem(final SimpleProblem p) {
		final int dim = p.dim;
		final DoubleMatrix1D wts = new DenseDoubleMatrix1D(dim);
		final double perplexity0 = perplexity(p.x,p.y,p.wt,wts);
		System.out.println("perplexity0: " + perplexity0);
		for(int ns=0;ns<10;++ns) {
			System.out.println();
			NewtonStep(p.x,p.y,p.wt,wts, true, 0.0);
			System.out.println(wts);
			final double perplexity1 = perplexity(p.x,p.y,p.wt,wts);
			System.out.println("perplexity" + (ns+1) + ": " + perplexity1);
			if(perplexity1>perplexity0) {
				System.out.println("break");
			}
		}
	}
	
	public static void bruteSolve2D(final SimpleProblem p) {
		final DoubleMatrix1D wts = new DenseDoubleMatrix1D(2);
		int nsteps = 100;
		double best = Double.POSITIVE_INFINITY;
		DoubleMatrix1D bestwts = null;
		final double range = 10.0;
		for(int xi=-nsteps;xi<=nsteps;++xi) {
			wts.set(0,range*xi/((double)nsteps));
			for(int yi=-nsteps;yi<=nsteps;++yi) {
				wts.set(1,range*yi/((double)nsteps));
				final double perplexity1 = perplexity(p.x,p.y,p.wt,wts);
				if((bestwts==null)||(perplexity1<best)) {
					best = perplexity1;
					bestwts = wts.copy();
				}
			}
		}
		System.out.println("best start: " + bestwts + "\t" + best);
		for(int i=0;i<wts.size();++i) {
			wts.set(i,bestwts.get(i));
		}
		for(int ns=0;ns<5;++ns) {
			NewtonStep(p.x,p.y,p.wt,wts, true, 0.0);
			System.out.println("wt:");
			System.out.println(wts);
			final double perplexity1 = perplexity(p.x,p.y,p.wt,wts);
			System.out.println("perplexity" + (ns+1) + ": " + perplexity1);
		}
	}
	

	private static final class Population {
		public SimpleProblem best = null;
		public double bestScore = Double.NEGATIVE_INFINITY;
		public final double[] pscore;
		public final SimpleProblem[] population;
		
		public Population(final Random rand, final int psize, final SimpleProblem[] starts) {
			pscore = new double[psize];
			population = new SimpleProblem[psize];
			final int nstart = starts.length;
			final double[] startScores = new double[nstart];
			for(int j=0;j<nstart;++j) {
				final SimpleProblem sj = starts[j];
				startScores[j] = scoreExample(sj.x,sj.y,sj.wt);
				if((null==best)||(startScores[j]>bestScore)) {
					best = sj;
					bestScore = startScores[j];
				}
			}
			for(int i=0;i<psize;++i) {
				final int vi = rand.nextInt(nstart);
				population[i] = starts[vi];
				pscore[i] = startScores[vi];
			}
		}
		
		public Population(final Random rand, final Population o, final int psize) {
			pscore = new double[psize];
			population = new SimpleProblem[psize];
			best = o.best;
			bestScore = o.bestScore;
			final int osize = o.population.length;
			for(int i=0;i<psize;++i) {
				final int vi = rand.nextInt(osize);
				population[i] = o.population[vi];
				pscore[i] = o.pscore[vi];
			}
		}
		
		public boolean show(final SimpleProblem p, final double score) {
			if((null==best)||(score>bestScore)) {
				best = p;
				bestScore = score;
				return true;
			}
			return false;
		}
	}
	
	private static class AnnealJob1 implements Runnable {
		public final int id;
		public final int psize;
		public final Random rand;
		public final Population shared;
		
		public AnnealJob1(final int id, final int psize, final Random rand, final Population shared) {
			this.id = id;
			this.psize = psize;
			this.rand = rand;
			this.shared = shared;
		}
		
		private void swap(final Population p) {
			synchronized(shared) {
				//System.out.println("Runnable " + id + " mixing into main population " + new Date());
				for(int oi=0;oi<psize;++oi) {
					if(rand.nextBoolean()) {
						final int ti = rand.nextInt(shared.population.length);
						final SimpleProblem or = p.population[oi];
						final double os = p.pscore[oi];
						p.population[oi] = shared.population[ti];
						p.pscore[oi] = shared.pscore[ti];
						shared.population[ti] = or;
						shared.pscore[ti] = os;
					}
				}
			}
		}

		protected double score(final SimpleProblem mi) {
			return scoreExample(mi.x,mi.y,mi.wt);
		}
		
		protected int nInserts(final double score) {
			final int nInserts = Math.max((int)Math.floor(10.0*score),1);
			return nInserts;
		}
		
		@Override
		public void run() {
			final Population p;
			synchronized(shared) {
				System.out.println("anneal Runnable " + id + " start " + new Date());
				p = new Population(rand,shared,psize);
			}
			for(int step=0;step<2*psize;++step) {
				final int di = rand.nextInt(psize);
				final SimpleProblem donor = p.population[di];
				final double dscore = p.pscore[di];
				final Set<SimpleProblem> mutations = SimpleProblem.mutations(donor);
				final SimpleProblem d2 = p.population[rand.nextInt(psize)];
				final Set<SimpleProblem> children = SimpleProblem.breed(donor,d2,rand);
				mutations.addAll(children);
				boolean record = false;
				for(final SimpleProblem mi: mutations) {
					final double scorem = score(mi);
					if(p.show(mi,scorem)) {
						record = true;
						synchronized(shared) {
							if(shared.show(mi,scorem)) {
								System.out.println("new record: " + p.bestScore + "\t" + p.best + "\t" + new Date());
							}
						}
					}
					final double ms = Math.max(scorem,0.5*(scorem+dscore)); // effective score (some credit from one parent)
					final int nInserts = nInserts(ms);
					for(int insi=0;insi<nInserts;++insi) {
						final int vi = rand.nextInt(psize);
						// 	number of insertions*worseodds < 1 to ensure progress
						if((ms>p.pscore[vi])||(rand.nextDouble()>0.9)) {
							p.population[vi] = mi;
							p.pscore[vi] = ms;
						}
					}
				}
				// mix into shared population 
				if(record||(step%(psize/2))==0) {
					swap(p);
				}
			}
			swap(p);
			synchronized(shared) {
				System.out.println("anneal Runnable " + id + " finish " + new Date());
			}
		}
	}
	
	private static class AnnealJob2 extends AnnealJob1 {
		public AnnealJob2(final int id, final int psize, final Random rand, final Population shared) {
			super(id,psize,rand,shared);
		}
		
		@Override
		protected double score(final SimpleProblem mi) {
			return scoreExample2(mi.x,mi.y,mi.wt);
		}
		
		@Override
		protected int nInserts(final double score) {
			if(score<1.0) {
				return Math.min(19,Math.max((int)Math.floor(-3.0*Math.log(1.0-score)),1));
			} else {
				return 20;
			}
		}

	}

	/**
	 * @param args
	 * @throws InterruptedException 
	 */
	public static void main1(String[] args) throws InterruptedException {
		//System.out.println("showing problem:");
		//showProblem(zeroSolnProblem,-6,6,-6,6);
		//showProblem(badZeroStartProblem,-12,-2,-12,-2);
		//final Set<SimpleProblem> starts = searchForProblem();
		final Set<SimpleProblem> starts = randProbs(2);
		System.out.println("start anneal");
		final Random rand = new Random(235235);
		final Population shared = new Population(new Random(rand.nextLong()),500000,starts.toArray(new SimpleProblem[starts.size()]));
		{
			final int njobs = 20;
			final int nparallel = 6;
			final ArrayBlockingQueue<Runnable> queue = new ArrayBlockingQueue<Runnable>(njobs+1);
			final ThreadPoolExecutor executor = new ThreadPoolExecutor(nparallel,nparallel,100,TimeUnit.SECONDS,queue);
			for(int i=0;i<njobs;++i) {
				executor.execute(new AnnealJob1(i,100000,new Random(rand.nextLong()),shared));
			}
			executor.shutdown();
			executor.awaitTermination(Long.MAX_VALUE,TimeUnit.SECONDS);
			System.out.println("done anneal1");
		}
		//workProblem(example2D);
		//bruteSolve(example2D);
	}

	public static void main(String[] args) {
		final SimpleProblem p = new SimpleProblem(
				new double[][] { 
						{ 1.0,  0},
						{ 1.0,	0.01},
						{ 1.0,	0.001},
				},
				new boolean[] {
						false,
						true,
						true,
				},
				new int[] {
						1,
						1,
						1,
				}
				);
		final int wtBound = 50;
		final boolean[] yvals = { false, true };
		final double[] xvals = { 0.01, 0.1, 1, 10, 100,  -0.01, -0.1, -1, -10, -100 };
		for(final double x1: xvals) {
			p.x[1][1] = x1;
			for(final double x2: xvals) {
				p.x[2][1] = x2;
				for(final boolean y0: yvals) {
					p.y[0] = y0;
					for(final boolean y1: yvals) {
						p.y[1] = y1;
						for(final boolean y2: yvals) {
							p.y[2] = y2;
							for(p.wt[0]=1;p.wt[0]<wtBound;++p.wt[0]) {
								for(p.wt[1]=1;p.wt[1]<wtBound;++p.wt[1]) {
									for(p.wt[2]=1;p.wt[2]<wtBound;++p.wt[2]) {
										final DoubleMatrix1D wts = new DenseDoubleMatrix1D(p.dim);
										final double perplexity0 = perplexity(p.x,p.y,p.wt,wts);
										//System.out.println("p0: " + perplexity0);
										NewtonStep(p.x,p.y,p.wt,wts, false, 0.0);		
										//System.out.println("w: " + wts);
										final double perplexity1 = perplexity(p.x,p.y,p.wt,wts);
										//System.out.println("p1: " + perplexity1);
										if(perplexity1>perplexity0) {
											System.out.println("break");
										}
									}
								}
							}
						}
					}
				}
			}
		}
	}
}
