using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TrailEvolutionModelling.EditorTools
{
    abstract class Tool
    {
        public bool IsActive { get; private set; }

        public event EventHandler OnBegin;
        public event EventHandler OnEnd;

        public void Begin()
        {
            if (IsActive)
                return;

            IsActive = true;
            OnBegin?.Invoke(this, EventArgs.Empty);
            BeginImpl();
        }

        public bool End()
        {
            if (!IsActive)
                return false;

            IsActive = false;
            EndImpl();
            OnEnd?.Invoke(this, EventArgs.Empty);
            return true;
        }

        protected abstract void BeginImpl();
        protected abstract void EndImpl();
    }
}
