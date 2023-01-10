//----------------------------------*-C++-*----------------------------------//
// Copyright 2021-2022 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/Collection.test.hh
//---------------------------------------------------------------------------//
#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/Types.hh"
#include "corecel/data/Collection.hh"

namespace celeritas
{
namespace test
{
//---------------------------------------------------------------------------//
// MOCK DATA
//---------------------------------------------------------------------------//

struct MockElement
{
    int atomic_number = 0;
    double atomic_mass;
};

using MockElementId = ItemId<MockElement>;

struct MockMaterial
{
    double number_density;
    ItemRange<MockElement> elements;
};

using MockMaterialId = ItemId<MockMaterial>;

template<Ownership W, MemSpace M>
struct MockParamsData
{
    //// DATA ////

    Collection<MockElement, W, M> elements;
    Collection<MockMaterial, W, M> materials;
    int max_element_components{};

    //// MEMBER FUNCTIONS ////

    //! Whether the object is in a valid state
    explicit operator bool() const
    {
        return !materials.empty() && max_element_components >= 0;
    }

    //! Assign from another set of collections
    template<Ownership W2, MemSpace M2>
    MockParamsData& operator=(MockParamsData<W2, M2> const& other)
    {
        CELER_EXPECT(other);
        elements = other.elements;
        materials = other.materials;
        max_element_components = other.max_element_components;
        return *this;
    }
};

//---------------------------------------------------------------------------//
/*!
 * Mock-up of a set of states.
 */
template<Ownership W, MemSpace M>
struct MockStateData
{
    //// DATA ////

    StateCollection<MockMaterialId, W, M> matid;

    //// MEMBER FUNCTIONS ////

    explicit CELER_FUNCTION operator bool() const { return !matid.empty(); }
    CELER_FUNCTION size_type size() const { return matid.size(); }

    //! Assign from another set of collections on the host
    template<Ownership W2, MemSpace M2>
    MockStateData& operator=(MockStateData<W2, M2>& other)
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
    using ParamsData = NativeCRef<MockParamsData>;
    using StateData = NativeRef<MockStateData>;

    CELER_FUNCTION MockTrackView(ParamsData const& params,
                                 StateData const& states,
                                 ThreadId tid)
        : params_(params), states_(states), thread_(tid)
    {
        CELER_EXPECT(thread_ < states_.size());
    }

    CELER_FUNCTION MockMaterialId matid() const
    {
        return states_.matid[thread_];
    }

    CELER_FUNCTION double number_density() const
    {
        return this->mat().number_density;
    }

    CELER_FUNCTION Span<MockElement const> elements() const
    {
        return params_.elements[this->mat().elements];
    }

  private:
    ParamsData const& params_;
    StateData const& states_;
    ThreadId thread_;

    CELER_FUNCTION MockMaterial const& mat() const
    {
        MockMaterialId id = this->matid();
        CELER_ASSERT(id < params_.materials.size());
        return params_.materials[id];
    }
};

//---------------------------------------------------------------------------//
// TESTING INTERFACE
//---------------------------------------------------------------------------//
//! Input data
struct CTestInput
{
    DeviceCRef<MockParamsData> params;
    DeviceRef<MockStateData> states;
    Span<double> result;
};

//---------------------------------------------------------------------------//
//! Run on device and return results
void col_cuda_test(CTestInput);

#if !CELER_USE_DEVICE
inline void col_cuda_test(CTestInput)
{
    CELER_NOT_CONFIGURED("CUDA or HIP");
}
#endif

//---------------------------------------------------------------------------//
}  // namespace test
}  // namespace celeritas
