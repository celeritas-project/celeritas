//----------------------------------*-C++-*----------------------------------//
// Copyright 2021 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file Pie.test.hh
//---------------------------------------------------------------------------//

#include "base/Pie.hh"
#include "base/Types.hh"

namespace celeritas_test
{
using celeritas::MemSpace;
using celeritas::Ownership;

//---------------------------------------------------------------------------//
// MOCK PIES
//---------------------------------------------------------------------------//

struct MockElement
{
    int    atomic_number = 0;
    double atomic_mass;
};

struct MockMaterial
{
    double                           number_density;
    celeritas::PieSlice<MockElement> elements;
};

template<Ownership W, MemSpace M>
struct MockParamsPies
{
    //// TYPES ////

    template<class T>
    using Pie = celeritas::Pie<T, W, M>;

    //// DATA ////

    celeritas::Pie<MockElement, W, M>  elements;
    celeritas::Pie<MockMaterial, W, M> materials;
    int                                max_element_components{};

    //// MEMBER FUNCTIONS ////

    //! Whether the object is in a valid state
    explicit operator bool() const
    {
        return !materials.empty() && max_element_components >= 0;
    }

    //! Assign from another set of pies
    template<Ownership W2, MemSpace M2>
    MockParamsPies& operator=(const MockParamsPies<W2, M2>& other)
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
struct MockStatePies
{
    //// TYPES ////

    template<class T>
    using Pie = celeritas::Pie<T, W, M>;

    //// DATA ////

    celeritas::Pie<int, W, M> matid;

    //// MEMBER FUNCTIONS ////

    explicit CELER_FUNCTION operator bool() const { return !matid.empty(); }
    CELER_FUNCTION celeritas::size_type size() const { return matid.size(); }

    // NOTE: no constructor from MockStatePies<W2, M2> means that calling
    //  MockStatePies<ref, host> foo = host_pies;
    // gives an ugly error message

    //! Assign from another set of pies on the host
    template<Ownership W2, MemSpace M2>
    MockStatePies& operator=(MockStatePies<W2, M2>& other)
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
        = MockParamsPies<Ownership::const_reference, MemSpace::native>;
    using StatePointers = MockStatePies<Ownership::reference, MemSpace::native>;
    using ThreadId      = celeritas::ThreadId;

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
    MockParamsPies<Ownership::const_reference, MemSpace::device> params;
    MockStatePies<Ownership::reference, MemSpace::device>        states;
    celeritas::Span<double>                                      result;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
void p_test(PTestInput);

//---------------------------------------------------------------------------//
} // namespace celeritas_test
