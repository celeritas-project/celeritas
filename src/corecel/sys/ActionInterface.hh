//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file corecel/sys/ActionInterface.hh
//---------------------------------------------------------------------------//
#pragma once

#include <string>
#include <string_view>

#include "corecel/cont/Span.hh"

#include "ThreadId.hh"

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Within-step ordering of explicit actions.
 *
 * Each "step iteration", wherein many tracks undergo a single step in
 * parallel, consists of an ordered series of actions. An action with an
 * earlier order always precedes an action with a later order.
 *
 * \sa StepActionInterface
 */
enum class StepActionOrder
{
    generate,  //!< Fill new track initializers
    start,  //!< Initialize tracks
    user_start,  //!< User initialization of new tracks
    sort_start,  //!< Sort track slots after initialization
    pre,  //!< Pre-step physics and setup
    user_pre,  //!< User actions for querying pre-step data
    sort_pre,  //!< Sort track slots after setting pre-step
    along,  //!< Along-step
    sort_along,  //!< Sort track slots after determining first step action
    pre_post,  //!< Discrete selection kernel
    sort_pre_post,  //! Sort track slots after selecting discrete interaction
    post,  //!< After step
    user_post,  //!< User actions after boundary crossing, collision
    end,  //!< Processing secondaries, including replacing primaries
    size_
};

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
 * The \c ActionInterface provides a clean virtual interface for
 * gathering metadata. The \c StepActionInterface provides additional
 * interfaces for launching kernels. The \c BeginRunActionInterface allows
 * actions to modify the state (or the class instance itself) at the beginning
 * of a stepping loop, and \c EndRunActionInterface allows actions to
 * gather and merge multiple state information at the end.
 *
 * Using multiple inheritance, you can create an action that inherits from
 * multiple of these classes. Note also that the function signatures are
 * similar to other high-level interfaces classes in Celeritas (e.g., \c
 * AuxParamsInterface, \c OutputInterface), so one "label" can be used to
 * satisfy multiple interfaces.
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
    virtual ~ActionInterface() noexcept = 0;

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
 * Interface that can modify the action's state.
 *
 * Most actions can modify \em only the local "state" being passed as an
 * argument. This one allows data to be allocated or initialized at the
 * beginning of the run.
 *
 * \todo Delete this to allow only stateless actions, since now
 * we have aux data? This will reduce overhead for virtual inheritance classes
 * too.
 */
class MutableActionInterface : public virtual ActionInterface
{
};

//---------------------------------------------------------------------------//
/*!
 * Traits class for actions that modify or access params/state.
 *
 * Using a single base class's typedefs is necessary for some compilers to
 * avoid an "ambiguous type alias" failure: "member 'CoreParams' found in
 * multiple base classes of different types". Note that adding this class to
 * the inheritance hierarchy (even multiple times) has no additional storage or
 * access cost.
 */
template<class P, template<MemSpace M> class S>
struct ActionTypeTraits
{
    //@{
    //! \name Type aliases
    using CoreParams = P;
    using CoreStateHost = S<MemSpace::host>;
    using CoreStateDevice = S<MemSpace::device>;
    using SpanCoreStateHost = Span<S<MemSpace::host>* const>;
    using SpanCoreStateDevice = Span<S<MemSpace::device>* const>;
    //@}
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
 *
 * \todo This is currently called once per each state on each CPU thread, and
 * it would be more sensible to call a single with all cooperative states.
 *
 * \warning Because this is called once per thread, the inheriting class is
 * responsible for thread safety (e.g. adding mutexes).
 */
template<class P, template<MemSpace M> class S>
class BeginRunActionInterface : public ActionTypeTraits<P, S>,
                                public MutableActionInterface
{
  public:
    //! Set host data at the beginning of a run
    virtual void begin_run(P const&, S<MemSpace::host>&) = 0;
    //! Set device data at the beginning of a run
    virtual void begin_run(P const&, S<MemSpace::device>&) = 0;
};

//---------------------------------------------------------------------------//
/*!
 * Interface for kernel actions in a stepping loop.
 *
 * \tparam P Core param class
 * \tparam S Core state class
 */
template<class P, template<MemSpace M> class S>
class StepActionInterface : public ActionTypeTraits<P, S>,
                            virtual public ActionInterface
{
  public:
    //! Dependency ordering of the action inside the step
    virtual StepActionOrder order() const = 0;
    //! Execute the action with host data
    virtual void step(P const&, S<MemSpace::host>&) const = 0;
    //! Execute the action with device data
    virtual void step(P const&, S<MemSpace::device>&) const = 0;
};

//---------------------------------------------------------------------------//
/*!
 * Concrete mixin utility class for managing an action.
 *
 * \par Example:
 * \code
  class KernellyPhysicsAction final : public CoreStepActionInterface,
                                      public ConcreteAction
  {
    public:
      // Construct with ID and label
      using ConcreteAction::ConcreteAction;

      void step(CoreParams const&, CoreStateHost&) const final;
      void step(CoreParams const&, CoreStateDevice&) const final;

      StepActionOrder order() const final { return StepActionOrder::post; }
  };

  class PlaceholderPhysicsAction final : public ConcreteAction
  {
    public:
      // Construct with ID and label
      using ConcreteAction::ConcreteAction;
  };
 * \endcode
 *
 * The \c noexcept declarations improve code generation for the
 * common use case of multiple inheritance.
 *
 * \note Use this class when multiple instances of the class may coexist in the
 * same stepping loop.
 */
class ConcreteAction : virtual public ActionInterface
{
  public:
    // Construct from ID, unique label
    ConcreteAction(ActionId id, std::string label) noexcept(!CELERITAS_DEBUG);

    // Construct from ID, unique label, and description
    ConcreteAction(ActionId id,
                   std::string label,
                   std::string description) noexcept(!CELERITAS_DEBUG);

    // Default destructor
    ~ConcreteAction() noexcept override;
    CELER_DELETE_COPY_MOVE(ConcreteAction);

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
/*!
 * Concrete utility class for managing an action with static strings.
 *
 * This is an implementation detail of \c StaticConcreteAction, but it can be
 * used manually for classes that inherit from multiple \c label methods (e.g.,
 * something that's both an action and has aux data) for which the mixin method
 * does not work.
 */
class StaticActionData
{
  public:
    // Construct from ID, unique label
    StaticActionData(ActionId id,
                     std::string_view label) noexcept(!CELERITAS_DEBUG);

    // Construct from ID, unique label, and description
    StaticActionData(ActionId id,
                     std::string_view label,
                     std::string_view description) noexcept(!CELERITAS_DEBUG);

    //! ID of this action for verification
    ActionId action_id() const { return id_; }

    //! Short label
    std::string_view label() const { return label_; }

    //! Descriptive label
    std::string_view description() const { return description_; }

  private:
    ActionId id_;
    std::string_view label_;
    std::string_view description_;
};

//---------------------------------------------------------------------------//
/*!
 * Concrete mixin utility class for managing an action with static strings.
 *
 * This is a typical use case for "singleton" actions where a maximum of one
 * can exist per stepping loop. The action ID still must be supplied at
 * runtime.
 *
 * \note Use this class when the label and description are compile-time
 * constants.
 */
class StaticConcreteAction : virtual public ActionInterface
{
  public:
    // Construct from ID, unique label
    StaticConcreteAction(ActionId id,
                         std::string_view label) noexcept(!CELERITAS_DEBUG);

    // Construct from ID, unique label, and description
    StaticConcreteAction(ActionId id,
                         std::string_view label,
                         std::string_view description) noexcept(!CELERITAS_DEBUG);

    // Default destructor
    ~StaticConcreteAction() override = default;
    CELER_DELETE_COPY_MOVE(StaticConcreteAction);

    //! ID of this action for verification
    ActionId action_id() const final { return sad_.action_id(); }

    //! Short label
    std::string_view label() const final { return sad_.label(); }

    //! Descriptive label
    std::string_view description() const final { return sad_.description(); }

  private:
    StaticActionData sad_;
};

//---------------------------------------------------------------------------//

//! Action order/ID tuple for comparison in sorting
struct OrderedAction
{
    StepActionOrder order;
    ActionId id;

    //! Ordering comparison for an action/ID
    CELER_CONSTEXPR_FUNCTION bool operator<(OrderedAction const& other) const
    {
        if (this->order < other.order)
            return true;
        if (this->order > other.order)
            return false;
        return this->id < other.id;
    }
};

//---------------------------------------------------------------------------//

// Get a string corresponding to a surface type
char const* to_cstring(StepActionOrder);

//---------------------------------------------------------------------------//
}  // namespace celeritas
