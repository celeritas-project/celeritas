//----------------------------------*-C++-*----------------------------------//
// Copyright 2022-2024 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celeritas/global/ActionInterface.hh
//! \todo Move non-core-specific code to corecel
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <string_view>

#include "celeritas/Types.hh"  // IWYU pragma: export

#include "CoreTrackDataFwd.hh"  // IWYU pragma: export

namespace celeritas
{
//---------------------------------------------------------------------------//
class CoreParams;
template<MemSpace M>
class CoreState;

namespace optical
{
class CoreParams;
template<MemSpace M>
class CoreState;
}  // namespace optical

//---------------------------------------------------------------------------//
/*!
 * Pure abstract interface for an action that could happen to a track.
 *
 * An action represents a possible state point or state change for a track.
 * Explicit actions (see \c StepActionInterface ) call kernels that change
 * the state (discrete processes, geometry boundary), and *implicit* actions
 * (which do not inherit from the explicit interface) are placeholders for
 * different reasons to pause the state or mark it for future modification
 * (range limitation, propagation loop limit).
 *
 * The \c ActionInterface provides a no-overhead virtual interface for
 * gathering metadata. The \c StepActionInterface provides additional
 * interfaces for launching kernels. The \c BeginRunActionInterface allows
 * actions to modify the state (or the class instance itself) at the beginning
 * of a stepping loop.
 *
 * Using multiple inheritance, you can create an action that inherits from
 * multiple of these classes.
 *
 * The \c label should be a brief lowercase hyphen-separated string, usually a
 * noun, with perhaps some sort of category being the first token.
 *
 * The \c description should be a verb phrase (and not have a title-cased
 * start).
 */
class ActionInterface
{
  public:
    // Default virtual destructor allows deletion by pointer-to-interface
    virtual ~ActionInterface();

    //! ID of this action for verification and ordering
    virtual ActionId action_id() const = 0;

    //! Short unique label of the action
    virtual std::string_view label() const = 0;

    //! Description of the action
    virtual std::string_view description() const = 0;

  protected:
    //!@{
    //! Allow construction and assignment only through daughter classes
    ActionInterface() = default;
    CELER_DEFAULT_COPY_MOVE(ActionInterface);
    //!@}
};

//---------------------------------------------------------------------------//
/*!
 * Interface for updating states at the beginning of the simulation.
 *
 * This is necessary for some classes that require deferred initialization
 * (either to the class itself or the state), for example because it needs the
 * number of total actions being run.
 *
 * If the class itself--rather than the state--needs initialization, try to
 * initialize in the constructor and avoid using this interface if possible.
 */
class BeginRunActionInterface : public virtual ActionInterface
{
  public:
    //@{
    //! \name Type aliases
    using CoreStateHost = CoreState<MemSpace::host>;
    using CoreStateDevice = CoreState<MemSpace::device>;
    //@}

  public:
    //! Set host data at the beginning of a run
    virtual void begin_run(CoreParams const&, CoreStateHost&) = 0;
    //! Set device data at the beginning of a run
    virtual void begin_run(CoreParams const&, CoreStateDevice&) = 0;
};

//---------------------------------------------------------------------------//
/*!
 * Interface for kernel actions in a stepping loop.
 *
 * \tparam P Core param class
 * \tparam S Core state class
 */
template<class P, template<MemSpace M> class S>
class StepActionInterface : public virtual ActionInterface
{
  public:
    //@{
    //! \name Type aliases
    using CoreStateHost = CoreState<MemSpace::host>;
    using CoreStateDevice = CoreState<MemSpace::device>;
    //@}

  public:
    //! Dependency ordering of the action
    virtual StepActionOrder order() const = 0;

    //! Execute the action with host data
    virtual void execute(CoreParams const&, CoreStateHost&) const = 0;

    //! Execute the action with device data
    virtual void execute(CoreParams const&, CoreStateDevice&) const = 0;
};

//! Action interface for core stepping loop
using CoreStepActionInterface = StepActionInterface<CoreParams, CoreState>;

//! Action interface for optical stepping loop
using OpticalStepActionInterface
    = StepActionInterface<optical::CoreParams, optical::CoreState>;

//---------------------------------------------------------------------------//
/*!
 * Concrete mixin utility class for managing an action.
 *
 * Example:
 * \code
  class KernellyPhysicsAction final : public CoreStepActionInterface,
                                      public ConcreteAction
  {
    public:
      // Construct with ID and label
      using ConcreteAction::ConcreteAction;

      void execute(CoreParams const&, CoreStateHost&) const final;
      void execute(CoreParams const&, CoreStateDevice&) const final;

      StepActionOrder order() const final { return StepActionOrder::post; }
  };

  class PlaceholderPhysicsAction final : public ConcreteAction
  {
    public:
      // Construct with ID and label
      using ConcreteAction::ConcreteAction;
  };
 * \endcode
 */
class ConcreteAction : public virtual ActionInterface
{
  public:
    // Construct from ID, unique label
    ConcreteAction(ActionId id, std::string label);

    // Construct from ID, unique label, and description
    ConcreteAction(ActionId id, std::string label, std::string description);

    //! ID of this action for verification
    ActionId action_id() const final { return id_; }

    //! Short label
    std::string_view label() const final { return label_; }

    //! Descriptive label
    std::string_view description() const final { return description_; }

  private:
    ActionId id_;
    std::string label_;
    std::string description_;
};

//---------------------------------------------------------------------------//
}  // namespace celeritas
