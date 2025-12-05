//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/vg/detail/VgNavStateWrapper.hh
//---------------------------------------------------------------------------//
#pragma once

#include <VecGeom/base/Transformation3D.h>
#include <VecGeom/navigation/NavStateFwd.h>

#include "corecel/Assert.hh"
#include "geocel/vg/VecgeomTypes.hh"
#if CELER_VGNAV == CELER_VGNAV_TUPLE
#    include <VecGeom/navigation/NavStateTuple.h>
#else
#    include <VecGeom/navigation/NavStateIndex.h>
#endif

#include "corecel/Macros.hh"

namespace celeritas
{
namespace detail
{
//---------------------------------------------------------------------------//
/*!
 * Wrap a local impl state in the same interface as a VecGeom NavState.
 *
 * Because the path navigator doesn't support the same functionality as
 * index/tuple, this class will never be used as part of the function runtime.
 */
class VgNavStateWrapper
{
  public:
    using VPlacedVolume = vecgeom::VPlacedVolume;
    using Transformation = vecgeom::Transformation3D;
#if CELER_VGNAV == CELER_VGNAV_TUPLE
    using VgNavState = vecgeom::NavStateTuple;
#else
    // Note that the type corresponds to the Celeritas type alias
    // VgNavStateImpl, but it is *only* used if VECEGOM_USE_NAVINDEX is defined
    using VgNavState = vecgeom::NavStateIndex;
#endif

    // Constructor takes reference to low-level state and boundary
    CELER_CONSTEXPR_FUNCTION
    VgNavStateWrapper(VgNavStateImpl& impl_state, VgBoundary& boundary);

    //! Default copy constructor: both will reference the same data
    VgNavStateWrapper(VgNavStateWrapper const&) = default;

    //! Assign state from another nav wrapper
    CELER_FUNCTION VgNavStateWrapper& operator=(VgNavStateWrapper const& other)
    {
        this->s_ = other.s_;
        this->b_ = other.b_;
        return *this;
    }

    //! Construct from an actual vecgeom nav state (used by ScopedVgNavState)
    CELER_FUNCTION VgNavStateWrapper& operator=(VgNavState const& other)
    {
#if CELER_VGNAV == CELER_VGNAV_TUPLE
        this->s_ = other.GetState();
#else
        this->s_ = other.GetNavIndex();
#endif
        this->b_ = to_vgboundary(other.IsOnBoundary());
        return *this;
    }

    //! Implicitly convert to a true vecgeom nav state
    CELER_FUNCTION
    operator VgNavState() const
    {
        VgNavState result{this->GetState()};
        result.SetBoundaryState(this->IsOnBoundary());
        return result;
    }

    CELER_FIF
    VgNavStateImpl const& GetState() const { return s_; }

    //! Debug print
    CELER_FIF
    void Print() const { VgNavState{*this}.Print(); }

    CELER_FIF
    void Push(VPlacedVolume const* v)
    {
#if VECGEOM_VERSION < 0x020000
        // VG1 returns value; VG2 modifies in place :(
        s_ =
#endif
            VgNavState::PushImpl(s_, v);
    }

    CELER_FIF
    void Pop()
    {
#if VECGEOM_VERSION < 0x020000
        // VG1 returns value; VG2 modifies in place :(
        s_ =
#endif
            VgNavState::PopImpl(s_);
    }

    CELER_FIF
    VPlacedVolume const* Top() const { return VgNavState::TopImpl(s_); }

    CELER_FIF
    unsigned char GetLevel() const { return VgNavState::GetLevelImpl(s_); }

    inline CELER_FUNCTION VPlacedVolume const* At(int level) const;

    inline CELER_FUNCTION void TopMatrix(Transformation& trans) const;

    CELER_FIF
    void Clear();

    CELER_FIF
    bool IsOutside() const;

    CELER_FIF
    bool IsOnBoundary() const { return to_bool(b_); }

    CELER_FIF
    void SetBoundaryState(bool b) { b_ = to_vgboundary(b); }

    CELER_FIF
    void CopyTo(VgNavStateWrapper* other) const { *other = *this; }

    CELER_FIF bool HasSamePathAsOther(VgNavStateWrapper const& other) const
    {
        return s_ == other.s_;
    }

    CELER_FIF
    void SetLastExited() {}

    CELER_FIF
    VPlacedVolume const* GetLastExited() { return nullptr; }

  private:
    VgNavStateImpl& s_;
    VgBoundary& b_;
};

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//

CELER_CONSTEXPR_FUNCTION
VgNavStateWrapper::VgNavStateWrapper(VgNavStateImpl& impl_state,
                                     VgBoundary& boundary)
    : s_{impl_state}, b_{boundary}
{
    CELER_EXPECT(CELER_VGNAV != CELER_VGNAV_PATH);
}

CELER_FIF
vecgeom::VPlacedVolume const* VgNavStateWrapper::At(int level) const
{
    CELER_EXPECT(level >= 0);
    // This value is 1 for navtuple (V2) or index (V1 only), but 2 for navindex
    // (V2)
    constexpr int parent_offset = CELER_VGNAV == CELER_VGNAV_TUPLE ? 1
                                  : VECGEOM_VERSION < 0x020000     ? 1
                                                                   : 2;
#if CELER_VGNAV == CELER_VGNAV_TUPLE
    auto index = VgNavState::GetNavTupleImpl(s_, level).Top();
#else
    auto index = VgNavState::GetNavIndexImpl(s_, level);
#endif
    return index != vg_outside_nav_index
               ? VgNavState::ToPlacedVolume(
                     VgNavState::NavInd(index + parent_offset))
               : nullptr;
}

CELER_FIF
void VgNavStateWrapper::TopMatrix(Transformation& trans) const
{
    VgNavState::TopMatrixImpl(s_, trans);
}

CELER_FIF
void VgNavStateWrapper::Clear()
{
#if CELER_VGNAV == CELER_VGNAV_TUPLE
    s_.Clear();
#else
    s_ = vg_outside_nav_index;
#endif
    this->SetBoundaryState(false);
}

CELER_FIF
bool VgNavStateWrapper::IsOutside() const
{
#if CELER_VGNAV == CELER_VGNAV_TUPLE
    return (s_.Top() == vg_outside_nav_index);
#else
    return s_ == vg_outside_nav_index;
#endif
}

//---------------------------------------------------------------------------//
}  // namespace detail
}  // namespace celeritas
