package com.winvector.lstep;

import java.util.Random;
import java.util.Set;
import java.util.TreeSet;

import com.winvector.anneal.AnnealAdapter;
import com.winvector.anneal.RunAnneal;

public class GSearch {
	/**
	 * @param args
	 * @throws InterruptedException 
	 */
	public static void main(String[] args) throws InterruptedException {
		final Set<SimpleProblem> starts = randProbs(2);
		final AnnealAdapter<SimpleProblem> pv = new ProblemVariations();
		final SimpleProblem found = RunAnneal.runAnneal(pv, starts);
		System.out.println("found: " + found);
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
}
