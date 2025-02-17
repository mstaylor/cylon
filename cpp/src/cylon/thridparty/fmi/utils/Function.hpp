/*
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef CYLON_FUNCTION_HPP
#define CYLON_FUNCTION_HPP

#include <functional>

namespace FMI::Utils {
    //! Small wrapper around an arbitrary C++ binary function with signature T(T,T), i.e. accepting two arguments of type T and returning one of type T.
    template<typename T>
    class Function {
    public:
        Function(std::function<T(T,T)> f, bool commutative, bool associative) : f(f), commutative(commutative), associative(associative) {}

        T operator()(T a, T b) const {
            return f(a, b);
        }

        //! User provided information about commutativity
        bool commutative;
        //! User provided information about associativity
        bool associative;
    private:
        std::function<T(T,T)> f;
    };
}

#endif //CYLON_FUNCTION_HPP
