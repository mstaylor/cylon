##
 # Licensed under the Apache License, Version 2.0 (the "License");
 # you may not use this file except in compliance with the License.
 # You may obtain a copy of the License at
 #
 # http://www.apache.org/licenses/LICENSE-2.0
 #
 # Unless required by applicable law or agreed to in writing, software
 # distributed under the License is distributed on an "AS IS" BASIS,
 # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 # See the License for the specific language governing permissions and
 # limitations under the License.
 ##

'''
Mapping Cylon C++ UCXConfigMapV Type with PyCylon CommType
'''

IF CYTHON_UCX & CYTHON_UCC:

    from pycylon.net.ucx_config_mapv cimport CUCXConfigMapV

    cpdef enum CommType:
        PORT = CUCXConfigMapV._PORT
        ADDRESS = CUCXConfigMapV._ADDRESS
