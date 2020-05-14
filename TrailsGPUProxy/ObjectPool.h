#pragma once
#include <functional>
#include <vector>
#include <gcroot.h>
#include "IResource.h"
#include "ResourceManager.h"
#include "ComputeNodesHost.h"

namespace TrailEvolutionModelling {
    namespace GPUProxy {

        template <typename T> class ObjectPool;

        template <typename T>
        struct PoolEntry {
            T* object;

            PoolEntry(T* object, ObjectPool<T>* pool);
            void ReturnToPool();
        private:
            ObjectPool<T>* pool;
        };



        using namespace System::Threading;

        template <typename T>
        class ObjectPool : public IResource {
            friend class ResourceManager;
            
        public:
            PoolEntry<T> Take();
            void Return(PoolEntry<T> entry);
            PoolEntry<T> Instantiate();

        protected:
            ObjectPool(int initialSize, std::function<T* ()> factory);
            ObjectPool(int initialSize, std::function<T* ()> factory,
                std::function<void(T*, ResourceManager&)> deleter);
            void Free(ResourceManager& resources) override;

        private:
            static void DefaultDeleter(T* object, ResourceManager& resources);

        private:
            std::vector<PoolEntry<T>> all;
            std::vector<PoolEntry<T>> available;
            std::function<T* ()> factory;
            std::function<void(T*, ResourceManager&)> deleter;
            gcroot<Object^> sync;
        };

        template<typename T>
        inline ObjectPool<T>::ObjectPool(int initialSize, std::function<T* ()> factory) 
            : ObjectPool(initialSize, factory, DefaultDeleter)
        {
        }

        template<typename T>
        inline ObjectPool<T>::ObjectPool(int initialSize, std::function<T* ()> factory,
            std::function<void(T*, ResourceManager&)> deleter)
            : factory(factory),
              deleter(deleter),
              sync(gcnew Object)
        {
            for(int i = 0; i < initialSize; i++) {
                PoolEntry entry = Instantiate();
                all.push_back(entry);
                available.push_back(entry);
            }
        }

        template<typename T>
        inline PoolEntry<T> ObjectPool<T>::Take() {
            Monitor::Enter(sync);
            try {
                if(available.empty()) {
                    PoolEntry<T> entry = Instantiate();
                    all.push_back(entry);
                    return entry;
                }
                else {
                    PoolEntry<T> entry = available.back();
                    available.pop_back();
                    return entry;
                }
            }
            finally {
                Monitor::Exit(sync);
            }
        }

        template<typename T>
        inline void ObjectPool<T>::Return(PoolEntry<T> entry) {
            Monitor::Enter(sync);
            try {
                available.push_back(entry);
            }
            finally {
                Monitor::Exit(sync);
            }
        }

        template<typename T>
        inline PoolEntry<T> ObjectPool<T>::Instantiate() {
            return PoolEntry<T>(factory(), this);
        }

        template<typename T>
        inline void ObjectPool<T>::Free(ResourceManager& resources) {
            for(auto entry : all) {
                deleter(entry.object, resources);
            }
        }

        template<typename T>
        inline void ObjectPool<T>::DefaultDeleter(T* object, ResourceManager& resources) {
            resources.Free(object);
        }





        template<typename T>
        inline PoolEntry<T>::PoolEntry(T* object, ObjectPool<T>* pool)
            : object(object), pool(pool) 
        {
        }

        template<typename T>
        inline void PoolEntry<T>::ReturnToPool() {
            pool->Return(*this);
        }

    }
}