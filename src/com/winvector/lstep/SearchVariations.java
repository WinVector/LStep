package com.winvector.lstep;

import cern.colt.matrix.DoubleMatrix1D;
import cern.colt.matrix.impl.DenseDoubleMatrix1D;

public class SearchVariations {

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
		long nfound = 0;
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
										final double perplexity0 = ScoreStep.perplexity(p.x,p.y,p.wt,wts);
										//System.out.println("p0: " + perplexity0);
										ScoreStep.NewtonStep(p.x,p.y,p.wt,wts, false, 0.0);		
										//System.out.println("w: " + wts);
										final double perplexity1 = ScoreStep.perplexity(p.x,p.y,p.wt,wts);
										//System.out.println("p1: " + perplexity1);
										if(perplexity1>perplexity0) {
											System.out.println("found: " + p);
											++nfound;
										}
									}
								}
							}
						}
					}
				}
			}
		}
		System.out.println("nfound: " + nfound);
	}

}
