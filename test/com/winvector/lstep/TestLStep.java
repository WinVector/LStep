package com.winvector.lstep;

import static org.junit.Assert.*;

import org.junit.Test;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;

public class TestLStep {
	@Test
	public void testSigmoid() {
		final int nSteps = 1000;
		for(int i=-nSteps;i<=nSteps;++i) {
			final double x = 10.0*(i/(double)nSteps);
			final double sx = ScoreStep.sigmoid(x);
			final double cx = ScoreStep.simpleSigmoid(x);
			assertTrue(Math.abs(sx-cx)<1.0e-6);
		}
	}
	
	@Test
	public void testLogSigmoid() {
		final int nSteps = 1000;
		for(int i=-nSteps;i<=nSteps;++i) {
			final double x = 10.0*(i/(double)nSteps);
			final double sx = ScoreStep.logSigmoid(x);
			final double cx = Math.log(ScoreStep.simpleSigmoid(x));
			assertTrue(Math.abs(sx-cx)<1.0e-6);
		}
	}

	@Test
	public void testLogOneMinusSigmoid() {
		final int nSteps = 1000;
		for(int i=-nSteps;i<=nSteps;++i) {
			final double x = 10.0*(i/(double)nSteps);
			final double sx = ScoreStep.logOneMinusSigmoid(x);
			final double cx = Math.log(1.0-ScoreStep.simpleSigmoid(x));
			assertTrue(Math.abs(sx-cx)<1.0e-6);
		}
	}
	
	@Test
	public void testExampleSoln() {
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
			ProblemVariations.NewtonStep(x,y,null,wts, false, 0.0);
		}
		final double[] expect = { 0.2415,       0.7573,       0.3530 };
		for(int i=0;i<expect.length;++i) {
			assertTrue(Math.abs(expect[i]-wts.get(i))<1.0e-4);
		}
	}
	

}
