package com.winvector.lstep;

import java.util.Map;
import java.util.TreeMap;

public final class SimpleProblem implements Comparable<SimpleProblem> {
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
	
	private static class Row implements Comparable<Row> {
		public double[] x;
		public boolean y;
		
		public Row(final double[] x, final boolean y) {
			this.x = x;
			this.y = y;
		}

		@Override
		public int compareTo(final Row o) {
			final int n = x.length;
			if(n!=o.x.length) {
				if(n>=o.x.length) {
					return 1;
				} else {
					return -1;
				}
			}
			for(int i=0;i<n;++i) {
				final double diff = x[i] - o.x[i];
				if(Math.abs(diff)>1.0e-8) {
					if(diff>=0) {
						return 1;
					} else {
						return -1;
					}
				}
			}
			if(y!=o.y) {
				if(y) {
					return 1;
				} else {
					return -1;
				}
			}
			return 0;
		}
		
		@Override
		public boolean equals(final Object o) {
			return compareTo((Row)o)==0;
		}
	}
	
	public String toString() {
		final StringBuilder b = new StringBuilder();
		final int dim = x[0].length;
		b.append("(");
		for(int j=0;j<dim;++j) {
			b.append(",x" + j + "=c(");
			for(final double[] xi: x) {
				b.append("," + xi[j]);
			}
			b.append(")");
		}
		b.append(",y=c(");
		for(final boolean yi: y) {
			b.append("," + (yi?"T":"F"));
		}
		b.append(")");
		if(null!=wt) {
			b.append(",w=c(");
			for(final int wi: wt) {
				b.append("," + wi);
			}
			b.append(")");
		}
		b.append(")");
		return b.toString().replaceAll("\\(,","(");
	}
	
	public static SimpleProblem cleanRep(final double[][] x, final boolean[] y, final int[] wt) {
		final Map<Row,Integer> wCount = new TreeMap<Row,Integer>();
		final int om = x.length;
		final int dim = x[0].length;
		for(int i=0;i<om;++i) {
			final int wti = (null==wt)?1:wt[i]; 
			if(wti>0) {
				final Row r = new Row(x[i],y[i]);
				Integer oc = wCount.get(r);
				if(null==oc) {
					oc = wti;
				} else {
					oc = oc + wti;
				}
				wCount.put(r,oc);
			}
		}
		final int nm = wCount.size();
		final double[][] nx = new double[nm][dim];
		final boolean[] ny = new boolean[nm];
		final int[] nw = new int[nm];
		int i = 0;
		for(final Map.Entry<Row,Integer> me: wCount.entrySet()) {
			final Row r = me.getKey();
			final int c = me.getValue();
			for(int j=0;j<dim;++j) {
				nx[i][j] = r.x[j];
			}
			ny[i] = r.y;
			nw[i] = c;
			++i;
		}
		return new SimpleProblem(nx,ny,nw);
	}

	/**
	 * assumes we are in cannonical form from cleanRep()
	 */
	@Override
	public int compareTo(final SimpleProblem o) {
		final int m = x.length;
		if(m!=o.x.length) {
			if(m>=o.x.length) {
				return 1;
			} else {
				return -1;
			}
		}
		final int dim = x[0].length;
		if(dim!=o.x[0].length) {
			if(dim>=o.x[0].length) {
				return 1;
			} else {
				return -1;
			}
		}
		for(int i=0;i<m;++i) {
			if(y[i]!=o.y[i]) {
				if(y[i]) {
					return 1;
				} else {
					return -1;
				}
			}
			if(wt[i]!=o.wt[i]) {
				if(wt[i]>=o.wt[i]) {
					return 1;
				} else {
					return -1;
				}
			}
			for(int j=0;j<dim;++j) {
				if(Math.abs(x[i][j]-o.x[i][j])>1.0e-8) {
					if(x[i][j]>=o.x[i][j]) {
						return 1;
					} else {
						return -1;
					}
				}
			}
		}
		return 0;
	}
	
	@Override
	public boolean equals(final Object o) {
		return compareTo((SimpleProblem)o)==0;
	}
}
