//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file geocel/VolumeVisitor.hh
//---------------------------------------------------------------------------//
#pragma once

#include <iterator>
#include <unordered_set>
#include <vector>

namespace celeritas
{
//---------------------------------------------------------------------------//
/*!
 * Interface for accessing the volume graph.
 *
 * \tparam V Lightweight volume reference
 * \tparam VI Lightweight volume instance reference
 * \tparam CVI Container of volume instance references
 *
 * Note that this helper class is a template interface specification, not a
 * required base class. Providing the type aliases and member functions is all
 * that's needed.
 */
template<class V, class VI, class CVI = std::vector<VI>>
class VolumeAccessorInterface
{
  public:
    //! A lightweight identifier for a volume: OpaqueId or pointer
    using VolumeRef = V;
    //! A lightweight identifier for a volume instance
    using VolumeInstanceRef = VI;
    //! Container of child volume instances
    using ContainerVolInstRef = CVI;

  public:
    //! Outgoing volume node from an instance
    virtual VolumeRef volume(VolumeInstanceRef parent) = 0;
    //! Outgoing instance nodes from a volume
    virtual ContainerVolInstRef children(VolumeRef parent) = 0;

  protected:
    ~VolumeAccessorInterface() = default;
};

//---------------------------------------------------------------------------//
/*!
 * Recursively walk through all unique volumes/instances.
 *
 * \tparam VA Helper class with the same signature as VolumeAccessor above.
 *
 * This class can be used for both Geant4 and VecGeom to give the same visiting
 * behavior across the two. The volume accessor should have the same signature
 * as \c VolumeAccessor above.
 *
 * The visitor function must have the signature
 * <code>bool(*)(Ref, int)</code>
 * where the return value indicates whether the volume's children should be
 * visited, \c Ref is either VolumeRef or \c VolumeInstanceRef , and the
 * integer is the depth of the volume being visited (the top element has depth
 * zero).
 *
 * By default this will visit all unique instances, i.e. every path in the
 * graph (the entire "touchable" hierarchy): this may be
 * very expensive! If it's desired to only visit single physical volumes, mark
 * them as visited using a set (see unit test for example) and return \c false
 * from the visitor to terminate the search path.
 */
template<class VA>
class VolumeVisitor
{
  public:
    using VolumeRef = typename VA::VolumeRef;
    using VolumeInstanceRef = typename VA::VolumeInstanceRef;

    //! Construct from accessor for obtaining volumes
    explicit VolumeVisitor(VA va) : accessor_(std::forward<VA>(va)) {}

    // Visit a volume
    template<class F>
    inline void operator()(F&& visit, VolumeRef top);

    // Visit a volume instance
    template<class F>
    inline void operator()(F&& visit, VolumeInstanceRef top);

  private:
    //// TYPES ////

    struct QueuedVolume
    {
        VolumeInstanceRef vi;
        int depth;
    };

    //// DATA ////

    VA accessor_;
    std::vector<QueuedVolume> queue_;

    //// HELPERS ////

    inline void add_children(VolumeInstanceRef vol, int depth);
    inline void add_children(VolumeRef vol, int depth);
};

//---------------------------------------------------------------------------//
/*!
 * Visit the first volume/instance encountered, once, depth-first.
 *
 * \tparam T VolumeRef or VolumeInstanceRef
 * \tparam F Visitor function \c void(*)(Vol)
 */
template<class T, class F>
class VisitVolumeOnce
{
  public:
    //! Construct with volume/depth visitor
    VisitVolumeOnce(F&& visit) : visit_impl_(std::forward<F>(visit)) {}

    //! Visit a single volume
    bool operator()(T v, int)
    {
        if (!visited_.insert(v).second)
        {
            // Already visited
            return false;
        }
        visit_impl_(v);
        return true;
    }

  private:
    F visit_impl_;
    std::unordered_set<T> visited_;
};

//! Return a wrapper for a visitor function to make the visit unique
template<class T, class F>
auto make_visit_volume_once(F&& visit)
{
    return VisitVolumeOnce<T, F>{std::forward<F>(visit)};
}

//---------------------------------------------------------------------------//
// INLINE DEFINITIONS
//---------------------------------------------------------------------------//
/*!
 * Visit all volume instance paths, depth-first.
 *
 * Future work: we could keep and return full paths instead of just the depth
 * if we wanted.
 */
template<class VA>
template<class F>
void VolumeVisitor<VA>::operator()(F&& visit, VolumeInstanceRef top)
{
    // Add the top volume to the queue
    queue_ = {{top, 0}};

    // Visit remaining children instances
    while (!queue_.empty())
    {
        // Pop volume off the queue
        QueuedVolume qv = queue_.back();
        queue_.pop_back();

        if (visit(qv.vi, qv.depth))
        {
            this->add_children(qv.vi, qv.depth);
        }
    }
}

//---------------------------------------------------------------------------//
/*!
 * Visit all volume instance paths, depth-first.
 *
 * Future work: we could keep and return full paths instead of just the depth
 * if we wanted.
 */
template<class VA>
template<class F>
void VolumeVisitor<VA>::operator()(F&& visit, VolumeRef top)
{
    auto visit_impl = [this, &visit](VolumeRef v, int depth) {
        if (visit(v, depth))
        {
            this->add_children(v, depth);
        }
    };

    // Visit top and add children
    visit_impl(top, 0);

    // Visit remaining children instances
    while (!queue_.empty())
    {
        // Pop volume off the queue
        QueuedVolume qv = queue_.back();
        queue_.pop_back();

        visit_impl(accessor_.volume(qv.vi), qv.depth);
    }
}

//---------------------------------------------------------------------------//
/*!
 * Add child instances from the current volume to the queue.
 *
 * \arg depth Depth of the given volume
 */
template<class VA>
void VolumeVisitor<VA>::add_children(VolumeInstanceRef vi, int depth)
{
    return this->add_children(accessor_.volume(vi), depth);
}

//---------------------------------------------------------------------------//
/*!
 * Add child instances from the current to the queue.
 *
 * \arg depth Depth of the given volume
 */
template<class VA>
void VolumeVisitor<VA>::add_children(VolumeRef vol, int depth)
{
    auto&& children = accessor_.children(vol);
    // Append children in *reverse* order since we pop back
    auto const rend = std::reverse_iterator(children.begin());
    for (auto iter = std::reverse_iterator(children.end()); iter != rend;
         ++iter)
    {
        queue_.push_back({*iter, depth + 1});
    }
}

//---------------------------------------------------------------------------//
}  // namespace celeritas
