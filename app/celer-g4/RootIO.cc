//------------------------------- -*- C++ -*- -------------------------------//
// Copyright Celeritas contributors: see top-level COPYRIGHT file for details
// SPDX-License-Identifier: (Apache-2.0 OR MIT)
//---------------------------------------------------------------------------//
//! \file celer-g4/RootIO.cc
//---------------------------------------------------------------------------//
#include "RootIO.hh"

#include <cstdio>
#include <regex>
#include <G4Event.hh>
#include <G4RunManager.hh>
#include <G4Threading.hh>
#include <TBranch.h>
#include <TFile.h>
#include <TObject.h>
#include <TROOT.h>
#include <TTree.h>

#include "corecel/Macros.hh"
#include "corecel/io/Logger.hh"
#include "geocel/GeantUtils.hh"
#include "celeritas/ext/RootFileManager.hh"
#include "accel/SetupOptions.hh"

#include "GlobalSetup.hh"
#include "SensitiveHit.hh"

#ifdef _WIN32
#    include <process.h>
#else
#    include <unistd.h>
#endif

namespace celeritas
{
namespace app
{
//---------------------------------------------------------------------------//
/*!
 * Create a ROOT output file for each worker thread in MT.
 */
RootIO::RootIO()
{
    CELER_VALIDATE(RootFileManager::use_root(),
                   << "cannot interface with ROOT (disabled by user "
                      "environment)");

    ROOT::EnableThreadSafety();

    file_name_ = std::regex_replace(
        GlobalSetup::Instance()->GetSetupOptions()->output_file,
        std::regex("\\.json$"),
        ".root");

    if (file_name_.empty())
    {
        file_name_ = "celer-g4.root";
    }

    if (file_name_ == "-")
    {
        file_name_ = "stdout-" + std::to_string(::getpid()) + ".root";
    }

    if (G4Threading::IsWorkerThread())
    {
        file_name_ += std::to_string(G4Threading::G4GetThreadId());
    }

    if (G4Threading::IsWorkerThread()
        || !G4Threading::IsMultithreadedApplication())
    {
        CELER_LOG_LOCAL(debug)
            << "Creating ROOT event output file at '" << file_name_ << "'";

        file_.reset(TFile::Open(file_name_.c_str(), "recreate"));
        CELER_VALIDATE(file_->IsOpen(), << "failed to open " << file_name_);
        tree_.reset(new TTree(
            this->TreeName(), "event_hits", this->SplitLevel(), file_.get()));
    }
}

//---------------------------------------------------------------------------//
/*!
 * Return the static thread local singleton instance.
 */
RootIO* RootIO::Instance()
{
    static G4ThreadLocal RootIO instance;
    return &instance;
}

//---------------------------------------------------------------------------//
/*!
 * Write sensitive hits to output in the form of EventData.
 * See celeritas/io/EventData.hh
 */
void RootIO::Write(G4Event const* event)
{
    auto* hit_cols = event->GetHCofThisEvent();
    if (!hit_cols)
    {
        return;
    }

    // Populate EventData using the collections of sensitive hits
    EventData event_data;
    event_data.hits.resize(detector_name_id_map_.size());

    event_data.event_id = event->GetEventID();
    for (auto i : celeritas::range(hit_cols->GetNumberOfCollections()))
    {
        // NOTE: Geant4@10.5 G4VHitsCollection::GetName is not const correct
        auto* hc_id = hit_cols->GetHC(i);
        std::vector<EventHitData> hits;
        hits.resize(hc_id->GetSize());

        for (auto j : celeritas::range(hc_id->GetSize()))
        {
            auto* hit_id = dynamic_cast<SensitiveHit*>(hc_id->GetHit(j));
            hits[j] = hit_id->data();
        }

        auto const iter = detector_name_id_map_.find(hc_id->GetName());
        CELER_ASSERT(iter != detector_name_id_map_.end());
        event_data.hits[iter->second] = std::move(hits);
    }

    this->WriteObject(&event_data);
}

//---------------------------------------------------------------------------//
/*!
 * Fill event tree with event data.
 *
 * \note `tree_->Fill()` will import the data from *all* existing TBranches. So
 * this code expects to have only one TBranch in this TTree.
 */
void RootIO::WriteObject(EventData* event_data)
{
    if (!event_branch_)
    {
        //! \todo Expose root buffer size as environment variable if needed
        int const root_buffer_size{128000};

        event_branch_ = tree_->Branch(
            "event", &event_data, root_buffer_size, this->SplitLevel());
    }
    else
    {
        // NOLINTNEXTLINE(bugprone-multi-level-implicit-pointer-conversion)
        event_branch_->SetAddress(&event_data);
    }

    tree_->Fill();
    event_branch_->ResetAddress();
}

//---------------------------------------------------------------------------//
/*!
 * Map sensitive detectors to contiguous IDs.
 */
void RootIO::AddSensitiveDetector(std::string name)
{
    auto&& [iter, inserted]
        = detector_name_id_map_.insert({std::move(name), ++detector_id_});
    CELER_ASSERT(inserted);
    CELER_DISCARD(iter);
}

//---------------------------------------------------------------------------//
/*!
 * Write and Close or Merge output.
 */
void RootIO::Close()
{
    CELER_EXPECT((file_ && file_->IsOpen())
                 || (G4Threading::IsMultithreadedApplication()
                     && G4Threading::IsMasterThread()));

    if (!G4Threading::IsMultithreadedApplication())
    {
        CELER_LOG(info) << "Writing hit ROOT output to " << file_name_;
        CELER_ASSERT(tree_);
        file_->Write("", TObject::kOverwrite);
    }
    else
    {
        if (G4Threading::IsMasterThread())
        {
            // Merge output file on the master thread
            this->Merge();
        }
        else
        {
            CELER_LOG(debug) << "Writing temporary local ROOT output";
            file_->Write("", TObject::kOverwrite);
        }
    }

    event_branch_ = nullptr;
    tree_.reset();
    file_.reset();
}

//---------------------------------------------------------------------------//
/*!
 * Merging output root files from multiple threads using TTree::MergeTrees.
 *
 * TODO: use TBufferMerger and follow the example described in the ROOT
 * tutorials/multicore/mt103_fillNtupleFromMultipleThreads.C which stores
 * TBuffer data in memory and writes 32MB compressed output concurrently.
 */
void RootIO::Merge()
{
    auto const nthreads = celeritas::get_geant_num_threads();
    std::vector<TFile*> files;
    std::vector<TTree*> trees;
    std::unique_ptr<TList> list(new TList);

    CELER_LOG(info) << "Merging hit root files from " << nthreads
                    << " threads into \"" << file_name_ << "\"";

    for (int i = 0; i < nthreads; ++i)
    {
        std::string file_name = file_name_ + std::to_string(i);
        files.push_back(TFile::Open(file_name.c_str()));
        trees.push_back(dynamic_cast<TTree*>(files[i]->Get(this->TreeName())));
        list->Add(trees[i]);

        if (i == nthreads - 1)
        {
            auto* file = TFile::Open(file_name_.c_str(), "recreate");
            CELER_VALIDATE(file->IsOpen(), << "failed to open " << file_name_);

            auto* tree = TTree::MergeTrees(list.get());
            tree->SetName(this->TreeName());

            // Store sensitive detector map branch
            this->StoreSdMap(file);

            // Write both the TFile and TTree meta-data
            file->Write();
            file->Close();
        }
        // Delete the merged file
        std::remove(file_name.c_str());
    }
}

//---------------------------------------------------------------------------//
/*!
 * Store TTree with sensitive detector names and their IDs (used by
 * EventData).
 */
void RootIO::StoreSdMap(TFile* file)
{
    CELER_EXPECT(file && file->IsOpen());

    auto tree = new TTree(
        "sensitive_detectors", "name_to_id", this->SplitLevel(), file);

    std::string name;
    unsigned int id;
    tree->Branch("name", &name);
    tree->Branch("id", &id);

    for (auto const& iter : detector_name_id_map_)
    {
        name = iter.first;
        id = iter.second;
        tree->Fill();
    }
}

//---------------------------------------------------------------------------//
}  // namespace app
}  // namespace celeritas
