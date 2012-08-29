package com.winvector.lstep;

public class MeanStep {

	private static double logit(final double q) {
		return Math.log(q/(1.0-q));
	}

	
	/**
	 * @param args
	 */
	public static void main(String[] args) {
		final double eps = 1.0e-6;
		for(double q = 0.5 + eps;q<1.0;q+=eps) {
			final double check = logit(q) - 4.0*(q-0.5);
			if(check<0) {
				System.out.println("q: " + q);
			}
			double x = 0;
			for(int i=0;i<10;++i) {
				final double p = ScoreStep.sigmoid(x);
				if(p>q+1.0e-7) {
					System.out.println("step(" + q + "," + i + ")= " + x + ", p(x)=" + p);
				}
				x = x + (q-p)/(p*(1-p));
			}
		}
	}


}
