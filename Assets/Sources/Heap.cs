using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace TrailEvolutionModelling
{
    public class Heap<T>
    {
        private List<T> data;
        private Comparer<T> comparer;
        private Action<T, int> setItemId;

        public int Count { get; private set; }

        public Heap(Comparer<T> comparer, Action<T, int> setItemId)
        {
            this.comparer = comparer ?? Comparer<T>.Default;
            this.setItemId = setItemId;

            data = new List<T>(300 * 300);
        }

        public void Add(T item)
        {
            data.Add(item);
            setItemId(item, Count);
            Count++;
            Up(Count - 1);
        }

        public T Pop()
        {
            T top = data[0];
            Count--;

            if (Count > 0)
            {
                T x = data[Count];
                data[0] = x;
                setItemId(x, 0);
                Down(0);
            }
            data.RemoveAt(Count);
            setItemId(top, -1);

            return top;
        }

        public T Peek() => data[0];

        public void UpdateItem(int pos)
        {
            Down(pos);
            Up(pos);
        }

        private void Up(int pos)
        {
            T item = data[pos];
            while (pos > 0)
            {
                int parent = (pos - 1) >> 1;
                T current = data[parent];
                if (comparer.Compare(item, current) >= 0) break;
                data[pos] = current;

                setItemId(current, pos);
                pos = parent;
            }

            data[pos] = item;
            setItemId(item, pos);
        }

        private void Down(int pos)
        {
            int halfLength = Count >> 1;
            T item = data[pos];

            while (pos < halfLength)
            {
                var left = (pos << 1) + 1;
                var right = left + 1;
                var best = data[left];

                if (right < Count && comparer.Compare(data[right], best) < 0)
                {
                    left = right;
                    best = data[right];
                }
                if (comparer.Compare(best, item) >= 0) break;

                data[pos] = best;
                setItemId(best, pos);
                pos = left;
            }

            data[pos] = item;
            setItemId(item, pos);
        }
    }
}