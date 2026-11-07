//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/data/AuxStateVec.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>
#include <type_traits>
#include <vector>

#include "corecel/Config.hh"

#include "corecel/Assert.hh"
#include "corecel/Macros.hh"
#include "corecel/sys/TypeDemangler.hh"

#include "AuxInterface.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
class AuxParamsRegistry;

//---------------------------------------------------------------------------//
/*!
 * Manage single-stream auxiliary state data.
 *
 * This class is constructed from a \c AuxParamsRegistry after the params are
 * completely added and while the state is being constructed (with its size,
 * etc.). The AuxId for an element of this class corresponds to the
 * AuxParamsRegistry.
 *
 * This class can be empty either by default or if the given auxiliary registry
 * doesn't have any entries.
 */
class AuxStateVec
{
  public:
    //@{
    //! \name Type aliases
    using UPState = std::unique_ptr<AuxStateInterface>;
    using UPConstState = std::unique_ptr<AuxStateInterface const>;
    //@}

  public:
    //! Create without any auxiliary data
    AuxStateVec() = default;

    // Create from params on a device/host stream
    AuxStateVec(AuxParamsRegistry const&, MemSpace, StreamId, size_type);

    // Allow moving; copying is prohibited due to unique pointers
    CELER_DEFAULT_MOVE_DELETE_COPY(AuxStateVec);

    ~AuxStateVec() = default;

    // Access auxiliary state interfaces
    inline AuxStateInterface& at(AuxId);
    inline AuxStateInterface const& at(AuxId) const;

    //! Get the number of defined states
    AuxId::size_type size() const { return states_.size(); }

  private:
    std::vector<UPState> states_;
};

//---------------------------------------------------------------------------//
// FREE FUNCTIONS
//---------------------------------------------------------------------------//
/*!
 * Get a mutable item from a state vector efficiently and safely.
 */
template<class S>
S& get(AuxStateVec& vec, AuxId auxid)
{
    static_assert(std::is_base_of_v<AuxStateInterface, S>);
    CELER_EXPECT(auxid < vec.size());
    auto* ptr = &vec.at(auxid);
    if constexpr (CELERITAS_DEBUG)
    {
        CELER_VALIDATE(
            dynamic_cast<S*>(ptr),
            << "Aux state id " << auxid.get() << " should have type "
            << TypeDemangler<AuxStateInterface>{}(*ptr)
            << " but was accessed with type " << TypeDemangler<S>{}());
    }
    return *static_cast<S*>(ptr);
}

//---------------------------------------------------------------------------//
/*!
 * Get a const item from a state vector efficiently and safely.
 */
template<class S>
S const& get(AuxStateVec const& vec, AuxId auxid)
{
    static_assert(std::is_base_of_v<AuxStateInterface, S>);
    CELER_EXPECT(auxid < vec.size());
    auto* ptr = &vec.at(auxid);
    if constexpr (CELERITAS_DEBUG)
    {
        CELER_VALIDATE(
            dynamic_cast<S const*>(ptr),
            << "Aux state id " << auxid.get() << " should have type "
            << TypeDemangler<AuxStateInterface>{}(*ptr)
            << " but was accessed with type " << TypeDemangler<S>{}());
    }
    return *static_cast<S const*>(ptr);
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Access a mutable auxiliary state interface for a given ID.
 */
AuxStateInterface& AuxStateVec::at(AuxId id)
{
    CELER_EXPECT(id < states_.size());
    return *states_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
/*!
 * Access a mutable auxiliary state interface for a given ID.
 */
AuxStateInterface const& AuxStateVec::at(AuxId id) const
{
    CELER_EXPECT(id < states_.size());
    return *states_[id.unchecked_get()];
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
