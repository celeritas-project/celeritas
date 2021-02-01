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
    celeritas::PoolRange<MockElement> elements;
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

template<Ownership W, MemSpace M>
struct MockStatePools
{
    template<class T>
    using Pool = celeritas::Pool<T, W, M>;

    celeritas::Pool<int, W, M> matid;

    celeritas::size_type size() const { return matid.size(); }

    //! Assign from another set of pools
    template<Ownership W2, MemSpace M2>
    MockStatePools& operator=(MockStatePools<W2, M2>& other)
    {
        matid = other.matid;
        return *this;
    }
};

class MockTrackView
{
  public:
    using ParamsPointers
        = MockParamsPools<Ownership::const_reference, MemSpace::native>;
    using StatePointers
        = MockStatePools<Ownership::reference, MemSpace::native>;
    using ThreadId = celeritas::ThreadId;

    CELER_FUNCTION MockTrackView(const ParamsPointers& params,
                                 const StatePointers&  state,
                                 ThreadId              tid)
        : params_(params), states_(state), thread_(tid)
    {
        CELER_EXPECT(thread_ < states_.size());
    }

    int matid() const { return states_.matid[thread_.get()]; }

    double number_density() const
    {
        int                 id = this->matid();
        const MockMaterial& m  = params_.materials[id];
        return m.number_density;
    }

  private:
    const ParamsPointers& params_;
    const StatePointers&  states_;
    ThreadId              thread_;
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
