package com.winvector.anneal;

import java.util.ArrayList;
import java.util.Collection;
import java.util.Date;
import java.util.Random;
import java.util.Set;
import java.util.concurrent.ArrayBlockingQueue;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;



public final class RunAnneal<T extends Comparable<T>> {
	private final Population<T> shared;
	
	public RunAnneal(final Population<T> shared) {
		this.shared = shared;
	}
	
	private class AnnealJob1 implements Runnable {
		public final int id;
		public final int psize;
		public final Random rand;
			
		public AnnealJob1(final int id, final int psize, final Random rand) {
			this.id = id;
			this.psize = psize;
			this.rand = rand;
		}
		
		private void swap(final Population<T> p) {
			synchronized(shared) {
				//System.out.println("Runnable " + id + " mixing into main population " + new Date());
				for(int oi=0;oi<psize;++oi) {
					if(rand.nextBoolean()) {
						final int ti = rand.nextInt(shared.population.size());
						final T or = p.population.get(oi);
						final double os = p.pscore[oi];
						p.population.set(oi,shared.population.get(ti));
						p.pscore[oi] = shared.pscore[ti];
						shared.population.set(ti,or);
						shared.pscore[ti] = os;
					}
				}
			}
		}

		protected double score(final T mi) {
			return shared.pv.scoreExample(mi);
		}
		
		
		@Override
		public void run() {
			final Population<T> p;
			synchronized(shared) {
				System.out.println("anneal Runnable " + id + " start " + new Date());
				p = new Population<T>(rand,shared,psize);
			}
			T state = null;
			double stateScore = 0.0;
			{
				final int dIndex = rand.nextInt(psize);
				state = p.population.get(dIndex);
				stateScore = p.pscore[dIndex];
			}
			final int totstep = 10*psize;
			double temperature = 1.0;
			for(int step=0;step<totstep;++step) {
				final Set<T> mutations = shared.pv.mutations(state,rand);
				{
					final T d2 = p.population.get(rand.nextInt(psize));
					final Set<T> children = shared.pv.breed(state,d2,rand);
					mutations.addAll(children);
					mutations.add(d2);
				}
				final ArrayList<T> goodC = new ArrayList<T>(mutations.size());
				final ArrayList<Double> goodS = new ArrayList<Double>(mutations.size());
				for(final T mi: mutations) {
					if(mi.compareTo(state)!=0) {
						final double scorem = score(mi);
						if(scorem>0.0) {
							if(p.show(mi,scorem)) {
								synchronized(shared) {
									System.out.println("new job record, job" + id + ": "+ p.bestScore + "\t" + p.best + "\t" + new Date());
									if(shared.show(mi,scorem)) {
										System.out.println("new global record, job " + id + ": "+ p.bestScore + "\t" + p.best + "\t" + new Date());
									}
								}
							}
							goodC.add(mi);
							goodS.add(scorem);
							final int vi = rand.nextInt(psize);
							if(mi.compareTo(p.population.get(vi))!=0) {
								final double pTrans = Math.min(1,Math.exp((scorem-p.pscore[vi])/temperature));
								if((mi.compareTo(p.population.get(vi))!=0)&&(rand.nextDouble()<=pTrans)) {
									p.population.set(vi,mi);
									p.pscore[vi] = scorem;
								}
							}
						}
					}
				}
				boolean moved = false;
				if(!goodC.isEmpty()) {
					final int nNeighbor = goodC.size();
					final double[] odds = new double[nNeighbor];
					double totodds = 0.0;
					for(int i=0;i<nNeighbor;++i) {
						final double oi = Math.min(1,Math.exp((goodS.get(i)-stateScore)/temperature));
						odds[i] = oi;
						totodds += oi;
					}
					final double pickV = rand.nextDouble()*Math.max(1.0,totodds);
					int picked = -1;
					{
						double sumpick = 0.0;
						for(int i=0;i<nNeighbor;++i) {
							sumpick += odds[i];
							if(sumpick>=pickV) {
								picked = i;
								break;
							}
						}
					}
					if((picked>=0)&(state.compareTo(goodC.get(picked))!=0)) {
						state = goodC.get(picked);
						stateScore = goodS.get(picked);
						moved = true;
					} 
				} else {
					final int dIndex = rand.nextInt(psize);
					state = p.population.get(dIndex);
					stateScore = p.pscore[dIndex];
				}
				if(!moved) {
					temperature = Math.min(10.0,1.1*temperature);
				} else {
					temperature = Math.max(1.e-5,0.98*temperature);
				}
				// mix into shared population 
				if(step%(psize/2)==0) {
					swap(p);
					synchronized(shared) {
						System.out.println(" job " + id + ": temperature " + temperature + "\t" + new Date());
					}
				}
			}
			swap(p);
			synchronized(shared) {
				System.out.println("anneal Runnable " + id + " finish " + new Date());
			}
		}

	}


	private AnnealJob1 newJob(final int id, final int psize, final Random rand) {
		return new AnnealJob1(id,psize,rand);
	}
	
	public static <T extends Comparable<T>> T runAnneal(final AnnealAdapter<T> pv, final Collection<T> starts, final int nparallel) throws InterruptedException {
		System.out.println("start anneal");
		final Random rand = new Random(235235);
		final Population<T> shared = new Population<T>(pv,new Random(rand.nextLong()),50000,new ArrayList<T>(starts));
		final RunAnneal<T> ra = new RunAnneal<T>(shared);
		final int njobs = 20;
		if(nparallel>1) {
			final ArrayBlockingQueue<Runnable> queue = new ArrayBlockingQueue<Runnable>(njobs+1);
			final ThreadPoolExecutor executor = new ThreadPoolExecutor(nparallel,nparallel,100,TimeUnit.SECONDS,queue);
			for(int i=0;i<njobs;++i) {
				executor.execute(ra.newJob(i,1000,new Random(rand.nextLong())));
			}
			executor.shutdown();
			executor.awaitTermination(Long.MAX_VALUE,TimeUnit.SECONDS);
		} else {
			for(int i=0;i<njobs;++i) {
				ra.newJob(i,100000,new Random(rand.nextLong())).run();
			}
		}
		System.out.println("done anneal1");
		return shared.best;
	}

}
