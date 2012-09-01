package com.winvector.anneal;

import java.util.ArrayList;
import java.util.Random;




public final class Population<T extends Comparable<T>> {
	public T best = null;
	public double bestScore = Double.NEGATIVE_INFINITY;
	public final double[] pscore;
	public final ArrayList<T> population;
	public final AnnealAdapter<T> pv;
	
	public Population(final AnnealAdapter<T> pv, final Random rand, final int psize, final ArrayList<T> starts) {
		this.pv = pv;
		pscore = new double[psize];
		population = new ArrayList<T>(psize);
		final int nstart = starts.size();
		final double[] startScores = new double[nstart];
		for(int j=0;j<nstart;++j) {
			final T sj = starts.get(j);
			startScores[j] = pv.scoreExample(sj);
			if((null==best)||(startScores[j]>bestScore)) {
				best = sj;
				bestScore = startScores[j];
			}
		}
		for(int i=0;i<psize;++i) {
			final int vi = rand.nextInt(nstart);
			population.add(starts.get(vi));
			pscore[i] = startScores[vi];
		}
	}
	
	public Population(final Random rand, final Population<T> o, final int psize) {
		pv = o.pv;
		pscore = new double[psize];
		population = new ArrayList<T>(psize);
		best = o.best;
		bestScore = o.bestScore;
		final int osize = o.population.size();
		for(int i=0;i<psize;++i) {
			final int vi = rand.nextInt(osize);
			population.add(o.population.get(vi));
			pscore[i] = o.pscore[vi];
		}
	}
	
	public boolean show(final T p, final double score) {
		if((null==best)||(score>bestScore)) {
			best = p;
			bestScore = score;
			return true;
		}
		return false;
	}
}