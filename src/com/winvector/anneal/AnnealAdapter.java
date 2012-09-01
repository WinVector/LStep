package com.winvector.anneal;

import java.util.Random;
import java.util.Set;


/**
 * adapter to help implement simulated annealing or genetic search
 * @author johnmount
 *
 * @param <T>
 */
public interface AnnealAdapter<T extends Comparable<T>> {

	Set<T> mutations(T p, Random rand);

	Set<T> breed(T p0, T p1, Random rand);

	double scoreExample(T p);

}
