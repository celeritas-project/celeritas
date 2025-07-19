//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/optical/gen/GeneratorAction.hh
//---------------------------------------------------------------------------//
#pragma once

#include <memory>

#include "corecel/Macros.hh"
#include "corecel/data/AuxInterface.hh"
#include "corecel/data/AuxStateVec.hh"
#include "celeritas/optical/action/ActionInterface.hh"
#include "celeritas/phys/GeneratorInterface.hh"

#include "GeneratorBase.hh"
#include "GeneratorData.hh"
#include "OffloadData.hh"

#include "detail/GeneratorTraits.hh"

namespace celeritas
{
class CoreParams;

namespace optical
{
class MaterialParams;

//---------------------------------------------------------------------------//
/*!
 * Generate photons from optical distribution data.
 *
 * This samples and initializes optical photons directly in a track slot in a
 * reproducible way.  Multiple threads may generate initializers from a single
 * distribution.
 */
template<GeneratorType G>
class GeneratorAction final : public GeneratorBase
{
  public:
    //!@{
    //! \name Type aliases
    using TraitsT = detail::GeneratorTraits<G>;
    template<Ownership W, MemSpace M>
    using Data = typename TraitsT::template Data<W, M>;
    using SPConstParams = std::shared_ptr<typename TraitsT::Params const>;
    using SPConstMaterial = std::shared_ptr<MaterialParams const>;
    //!@}

    //! Generator input data
    struct Input
    {
        SPConstMaterial material;
        SPConstParams shared;
        size_type capacity{};

        explicit operator bool() const
        {
            return material && shared && capacity > 0;
        }
    };

  public:
    // Construct and add to core params
    static std::shared_ptr<GeneratorAction>
    make_and_insert(::celeritas::CoreParams const&, CoreParams const&, Input&&);

    // Construct with action ID, data IDs, and optical properties
    GeneratorAction(ActionId, AuxId, GeneratorId, Input&&);

    //!@{
    //! \name Aux interface

    // Build state data for a stream
    UPState create_state(MemSpace, StreamId, size_type) const final;
    //!@}

    //!@{
    //! \name StepAction interface

    // Launch kernel with host data
    void step(CoreParams const&, CoreStateHost&) const final;
    // Launch kernel with device data
    void step(CoreParams const&, CoreStateDevice&) const final;
    //!@}

  private:
    //// DATA ////

    Input data_;

    //// HELPER FUNCTIONS ////

    template<MemSpace M>
    void step_impl(CoreParams const&, CoreState<M>&) const;

    void generate(CoreParams const&, CoreStateHost&) const;
    void generate(CoreParams const&, CoreStateDevice&) const;
};

//---------------------------------------------------------------------------//
// EXPLICIT INSTANTIATION
//---------------------------------------------------------------------------//

extern template class GeneratorAction<GeneratorType::cherenkov>;
extern template class GeneratorAction<GeneratorType::scintillation>;

//---------------------------------------------------------------------------//
}  // namespace optical
}  // namespace celeritas
