package com.winvector.lstep;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.DoubleMatrix2D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix2D;
import cern.colt.matrix.linalg.Algebra;


/**
 * quick demonstration of Newton-Raphson divergence on logistic regression based on a quick implementation of
 * http://www.win-vector.com/blog/2011/09/the-simpler-derivation-of-logistic-regression/
 * @author johnmount
 *
 */
public final class ScoreStep {

	private static double dot(final DoubleMatrix1D a, final double[] b) {
		final int n = b.length;
		double r = 0.0;
		for(int j=0;j<n;++j) {
			r += a.get(j)*b[j];
		}
		return r;
	}
	
	private static double sigmoid(final double x) {
		return 1.0/(1.0 + Math.exp(-x));
	}
	
	/**
	 * @param x
	 * @param y
	 * @param update control if step is taken and wts are updated (inefficient way to compute score, but useful for debugging)
	 * @param wts (altered if update is true)
	 * @return perplexity of input wts
	 */
	private static final double NewtonStep(final double[][] x, final boolean[] y, final boolean update, final DoubleMatrix1D wts) {
		final int nDat = x.length;
		final int dim = x[0].length;
		final DoubleMatrix2D m = new DenseDoubleMatrix2D(dim,dim);
		final DoubleMatrix2D v = new DenseDoubleMatrix2D(dim,1);
		double perplexity = 0.0;
		for(int i=0;i<nDat;++i) {
			final double[] xi = x[i];
			final double pi = sigmoid(dot(wts,xi));
			final double mwt = pi*(1.0-pi);
			final double vwt = (y[i]?1.0:0.0) - pi;
			if(y[i]) {
				perplexity -= Math.log(pi);
			} else {
				perplexity -= Math.log(1.0-pi);
			}
			for(int j=0;j<dim;++j) {
				v.set(j,0,v.get(j,0) + vwt*xi[j]);
				for(int k=0;k<dim;++k) {
					m.set(j,k,m.get(j,k) + mwt*xi[j]*xi[k]);
				}
			}
		}
		if(update) {
			final DoubleMatrix2D delta = Algebra.DEFAULT.solve(m, v);
			for(int j=0;j<dim;++j) {
				wts.set(j,wts.get(j)+delta.get(j,0));
			}
		}
		return perplexity;
	}
	
	public static void showCorrectness() {
		/**
		 * From: http://www.win-vector.com/blog/2012/08/what-does-a-generalized-linear-model-do/
		 * > d <- read.table(file='http://www.win-vector.com/dfiles/glmLoss/dGLMdat.csv',
		 * +     header=T,sep=',')
		 * > d
		 *             x1          x2     y
		 * 1   0.09793624 -0.50020073 FALSE
		 * 2  -0.54933361  0.00834841  TRUE
		 * 3   0.18499020 -0.79325364  TRUE
		 * 4   0.58316450  2.06501637  TRUE
		 * 5   0.09607855  0.42724062  TRUE
		 * 6  -0.44772937  0.23267758 FALSE
		 * 7   1.24981165 -0.24492799  TRUE
		 * 8   0.13378532 -0.21985529  TRUE
		 * 9   0.41987141 -0.63677825 FALSE
		 * 10  1.28558918  1.37708143 FALSE
		 * 11  0.32590303  0.90813181  TRUE
		 * 12  0.01148262 -1.35426485 FALSE
		 * 13 -0.98502686  1.85317024  TRUE
		 * 14 -0.23017795 -0.06923035 FALSE
		 * 15  1.29606888 -0.80930538  TRUE
		 * 16  0.31286797  0.21319610  TRUE
		 * 17  0.03766960 -1.13314348  TRUE
		 * 18  0.03662855  0.67440240 FALSE
		 * 19  1.62032558 -0.57165979  TRUE
		 * 20 -0.63236983 -0.30736577 FALSE
		 * > m <- glm(y~x1+x2,data=d,family=binomial(link='logit'))
		 * > m
		 * 
		 * Call:  glm(formula = y ~ x1 + x2, family = binomial(link = "logit"), 
		 *     data = d)
		 * 
		 * Coefficients:
		 * (Intercept)           x1           x2  
		 *      0.2415       0.7573       0.3530  
		 * 
		 * Degrees of Freedom: 19 Total (i.e. Null);  17 Residual
		 * Null perplexity:	    26.92 
		 * Residual perplexity: 25.56 	AIC: 31.56 
		 * 
		 **/


		final double[][] x = { 
				{ 1.0,    0.09793624, -0.50020073 },
				{ 1.0,   -0.54933361,  0.00834841  },
				{ 1.0,    0.18499020, -0.79325364  },
				{ 1.0,    0.58316450,  2.06501637  },
				{ 1.0,    0.09607855,  0.42724062  },
				{ 1.0,   -0.44772937,  0.23267758 },
				{ 1.0,    1.24981165, -0.24492799  },
				{ 1.0,    0.13378532, -0.21985529  },
				{ 1.0,    0.41987141, -0.63677825 },
				{ 1.0,   1.28558918,  1.37708143 },
				{ 1.0,   0.32590303,  0.90813181  },
				{ 1.0,   0.01148262, -1.35426485 },
				{ 1.0,  -0.98502686,  1.85317024  },
				{ 1.0,  -0.23017795, -0.06923035 },
				{ 1.0,   1.29606888, -0.80930538  },
				{ 1.0,   0.31286797,  0.21319610  },
				{ 1.0,   0.03766960, -1.13314348  },
				{ 1.0,   0.03662855,  0.67440240 },
				{ 1.0,   1.62032558, -0.57165979  },
				{ 1.0,  -0.63236983, -0.30736577 },
		};
		final boolean[] y = {
				false,
				true,
				true,
				true,
				true,
				false,
				true,
				true,
				false,
				false,
				true,
				false,
				true,
				false,
				true,
				true,
				true,
				false,
				true,
				false,
		};
		final int dim = x[0].length;
		final DoubleMatrix1D wts = new DenseDoubleMatrix1D(dim);
		for(int step=0;step<10;++step) {
			NewtonStep(x,y,true,wts);
		}
		final double[] expect = { 0.2415,       0.7573,       0.3530 };
		double maxAbsDiff = 0.0;
		for(int i=0;i<expect.length;++i) {
			maxAbsDiff = Math.max(maxAbsDiff,Math.abs(expect[i]-wts.get(i)));
		}
		System.out.println("max abs diff: " + maxAbsDiff);
	}
	

	
	public static void showProblem() {
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
		// crummy model- but best soln at wts = 0
		final double[][] x = { 
				{ 1, 1 },
				{ 1, 0 },
				{ 1, 1 },
				{ 1, 0 }
		};
		final boolean[] y = {
				true,
				true,
				false,
				false
		};
		final int dim = x[0].length;
		final int nPts = 100;
		final double r = 6.0;
		final String sep = "\t";
		System.out.println("" + "wC" + sep + "wX" + sep + "perplexity0" + sep + "perplexity1" + sep + "decrease" + sep + "increase");
		for(int xi=-nPts;xi<=nPts;++xi) {
			final double w0 = xi*r/((double)nPts);
			for(int yi=-nPts;yi<=nPts;++yi) {
				final double w1 = yi*r/((double)nPts);
				final DoubleMatrix1D wts = new DenseDoubleMatrix1D(dim);
				wts.set(0,w0);
				wts.set(1,w1);
				final double perplexity0 = NewtonStep(x,y,true,wts);
				final double perplexity1 = NewtonStep(x,y,false,wts);
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
	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		//System.out.println("showing correctnes:");
		//showCorrectness();
		//System.out.println("showing problem:");
		showProblem();
	}

}
