package com.winvector.lstep;

public final class SimpleProblem {
	public final int dim;
	public final int nrow;
	public final double[][] x;
	public final boolean[] y;
	public final int[] wt;
	
	public SimpleProblem(final double[][] x,
			final boolean[] y,
			final int[] wt) {
		this.x = x;
		this.y = y;
		dim = x[0].length;
		nrow = x.length;
		if(null!=wt) {
			this.wt = wt;
		} else {
			this.wt = new int[nrow];
			for(int i=0;i<nrow;++i) {
				this.wt[i] = 1;
			}
		}
	}
}
