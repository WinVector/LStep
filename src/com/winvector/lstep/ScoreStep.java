package com.winvector.lstep;

import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;


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

	static double dot(final DoubleMatrix1D a, final double[] b) {
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
	private static SimpleProblem zeroSolnProblem = new SimpleProblem(
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
				final double perplexity0 = ProblemVariations.perplexity(prob.x,prob.y,prob.wt,wts);
				ProblemVariations.NewtonStep(prob.x,prob.y,prob.wt,wts, false, 0.0);
				final double perplexity1 = ProblemVariations.perplexity(prob.x,prob.y,prob.wt,wts);
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
			final double score = ProblemVariations.scoreExample(x, y, wt);
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


	
	private static final SimpleProblem example2D = new SimpleProblem(
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
		final double perplexity0 = ProblemVariations.perplexity(p.x,p.y,p.wt,wts);
		System.out.println("perplexity0: " + perplexity0);
		for(int ns=0;ns<10;++ns) {
			System.out.println();
			ProblemVariations.NewtonStep(p.x,p.y,p.wt,wts, true, 0.0);
			System.out.println(wts);
			final double perplexity1 = ProblemVariations.perplexity(p.x,p.y,p.wt,wts);
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
				final double perplexity1 = ProblemVariations.perplexity(p.x,p.y,p.wt,wts);
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
			ProblemVariations.NewtonStep(p.x,p.y,p.wt,wts, true, 0.0);
			System.out.println("wt:");
			System.out.println(wts);
			final double perplexity1 = ProblemVariations.perplexity(p.x,p.y,p.wt,wts);
			System.out.println("perplexity" + (ns+1) + ": " + perplexity1);
		}
	}
	


	

	/**
	 * @param args
	 * @throws InterruptedException 
	 */
	public static void main(String[] args) throws InterruptedException {
		// comment/un-comment various bits to get different runs
		
		// graph zeroSolnProblem
		showProblem(zeroSolnProblem,-6,6,-6,6);
		
		// graph badZeroStartProblem
		showProblem(badZeroStartProblem,-12,-2,-12,-2);
		
		// find some more baddies
		// can also try GSearch.main() 
		searchForProblem();
		
		// show divergencen
		workProblem(example2D);
		// show existence of a bounded solution
		bruteSolve2D(example2D);
	}
}
