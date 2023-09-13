
PREFIX=/home/alessandro/workspace/faiss

export PYTHONPATH=$PREFIX/build/faiss/python/build/lib/

swig -c++ -python  -I$PREFIX bow_id_selector.swig
g++ -shared -O3 -g -fPIC bow_id_selector_wrap.cxx -o _bow_id_selector.so  -I $( python3 -c "import distutils.sysconfig ; print(distutils.sysconfig.get_python_inc())" )    $PREFIX/build/faiss/libfaiss_avx2.so -I$PREFIX

