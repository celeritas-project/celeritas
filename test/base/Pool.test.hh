//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Pool.test.hh
//---------------------------------------------------------------------------//

#include "base/Pool.hh"

namespace celeritas_test
{
using celeritas::MemSpace;
using celeritas::Ownership;

//---------------------------------------------------------------------------//
// MOCK POOLS
//---------------------------------------------------------------------------//

struct MockElement
{
    int    atomic_number = 0;
    double atomic_mass;
};

struct MockMaterial
{
    double                           number_density;
    celeritas::PoolSpan<MockElement> elements;
};

template<Ownership W, MemSpace M>
struct MockParamsPools
{
    template<class T>
    using Pool = celeritas::Pool<T, W, M>;

    celeritas::Pool<MockElement, W, M>  elements;
    celeritas::Pool<MockMaterial, W, M> materials;
    int                                 max_element_components;

    //! Assign from another set of pools
    template<Ownership W2, MemSpace M2>
    MockParamsPools& operator=(const MockParamsPools<W2, M2>& other)
    {
        elements               = other.elements;
        materials              = other.materials;
        max_element_components = other.max_element_components;
        return *this;
    }
};

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct PTestInput
{
    int num_threads;
};

//---------------------------------------------------------------------------//
//! Output results
struct PTestOutput
{
};

//---------------------------------------------------------------------------//
//! Run on device and return results
PTestOutput p_test(PTestInput);

//---------------------------------------------------------------------------//
} // namespace celeritas_test
