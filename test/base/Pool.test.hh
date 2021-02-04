//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Pool.test.hh
//---------------------------------------------------------------------------//

#include "base/Pool.hh"
#include "base/Types.hh"

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
    celeritas::PoolSlice<MockElement> elements;
};

template<Ownership W, MemSpace M>
struct MockParamsPools
{
    //// TYPES ////

    template<class T>
    using Pool = celeritas::Pool<T, W, M>;

    //// DATA ////

    celeritas::Pool<MockElement, W, M>  elements;
    celeritas::Pool<MockMaterial, W, M> materials;
    int                                 max_element_components{};

    //// MEMBER FUNCTIONS ////

    //! Whether the object is in a valid state
    explicit operator bool() const
    {
        return !materials.empty() && max_element_components >= 0;
    }

    //! Assign from another set of pools
    template<Ownership W2, MemSpace M2>
    MockParamsPools& operator=(const MockParamsPools<W2, M2>& other)
    {
        CELER_EXPECT(other);
        elements               = other.elements;
        materials              = other.materials;
        max_element_components = other.max_element_components;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Mock-up of a set of states.
 */
template<Ownership W, MemSpace M>
struct MockStatePools
{
    //// TYPES ////

    template<class T>
    using Pool = celeritas::Pool<T, W, M>;

    //// DATA ////

    celeritas::Pool<int, W, M> matid;

    //// MEMBER FUNCTIONS ////

    explicit CELER_FUNCTION operator bool() const { return !matid.empty(); }
    CELER_FUNCTION celeritas::size_type size() const { return matid.size(); }

    // NOTE: no constructor from MockStatePools<W2, M2> means that calling
    //  MockStatePools<ref, host> foo = host_pools;
    // gives an ugly error message

    //! Assign from another set of pools on the host
    template<Ownership W2, MemSpace M2>
    MockStatePools& operator=(MockStatePools<W2, M2>& other)
    {
        CELER_EXPECT(other);
        matid = other.matid;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Mock-up of a track view.
 */
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

    CELER_FUNCTION int matid() const { return states_.matid[thread_.get()]; }

    CELER_FUNCTION double number_density() const
    {
        return this->mat().number_density;
    }

    CELER_FUNCTION celeritas::Span<const MockElement> elements() const
    {
        return params_.elements[this->mat().elements];
    }

  private:
    const ParamsPointers& params_;
    const StatePointers&  states_;
    ThreadId              thread_;

    CELER_FUNCTION const MockMaterial& mat() const
    {
        int id = this->matid();
        CELER_ASSERT(id < int(params_.materials.size()));
        return params_.materials[id];
    }
};

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct PTestInput
{
    MockParamsPools<Ownership::const_reference, MemSpace::device> params;
    MockStatePools<Ownership::reference, MemSpace::device>        states;
    celeritas::Span<double>                                       result;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
void p_test(PTestInput);

//---------------------------------------------------------------------------//
} // namespace celeritas_test
