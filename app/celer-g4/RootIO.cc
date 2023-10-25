//----------------------------------*-C++-*----------------------------------//
// Copyright 2023 UT-Battelle, LLC, and other Celeritas developers.
// See the top-level COPYRIGHT file for details.
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
#include "celeritas/ext/GeantSetup.hh"
#include "accel/ExceptionConverter.hh"
#include "accel/SetupOptions.hh"

#include "GlobalSetup.hh"
#include "SensitiveHit.hh"

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
    ROOT::EnableThreadSafety();

    file_name_ = std::regex_replace(
        GlobalSetup::Instance()->GetSetupOptions()->output_file,
        std::regex("\\.json$"),
        ".root");

    if (file_name_.empty())
    {
        file_name_ = "celer-g4.root";
    }

    if (G4Threading::IsWorkerThread())
    {
        file_name_ += std::to_string(G4Threading::G4GetThreadId());
    }

    if (G4Threading::IsWorkerThread()
        || !G4Threading::IsMultithreadedApplication())
    {
        CELER_LOG_LOCAL(info)
            << "Creating ROOT event output file at '" << file_name_ << "'";

        file_.reset(TFile::Open(file_name_.c_str(), "recreate"));
        CELER_VALIDATE(file_->IsOpen(), << "failed to open " << file_name_);
        tree_.reset(new TTree(this->TreeName(),
                              this->TreeName(),
                              this->SplitLevel(),
                              file_.get()));
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
    G4HCofThisEvent* hce = event->GetHCofThisEvent();
    if (hce == nullptr)
    {
        return;
    }

    // Write steps and collection of hits
    EventData event_data;
    event_data.event_id = event->GetEventID();
    for (int i = 0; i < hce->GetNumberOfCollections(); i++)
    {
        std::vector<HitData> hits;

        G4VHitsCollection* hc = hce->GetHC(i);
        for (std::size_t j = 0; j < hc->GetSize(); ++j)
        {
            auto* sd_hit = dynamic_cast<SensitiveHit*>(hc->GetHit(j));
            auto const& result = sd_hit->data();
            event_data.steps.push_back(result.step);
            hits.push_back(result.hit);
        }
        auto iter = detector_name_id_map_.find(hc->GetName());
        CELER_ASSERT(iter != detector_name_id_map_.end());
        event_data.hits.insert({iter->second, std::move(hits)});
    }

    this->WriteObject(&event_data);
}

//---------------------------------------------------------------------------//
/*!
 * Fill event tree with event data.
 */
void RootIO::WriteObject(EventData* event_data)
{
    if (!event_branch_)
    {
        event_branch_
            = tree_->Branch("event",
                            &event_data,
                            GlobalSetup::Instance()->GetRootBufferSize(),
                            this->SplitLevel());
    }
    else
    {
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
    auto iter = detector_name_id_map_.find(name);
    if (iter == detector_name_id_map_.end())
    {
        detector_name_id_map_.insert({name, ++detector_id_});
    }
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
    auto const nthreads = get_num_threads(*G4RunManager::GetRunManager());
    std::vector<TFile*> files;
    std::vector<TTree*> trees;
    std::unique_ptr<TList> list(new TList);

    CELER_LOG_LOCAL(info) << "Merging hit root files from " << nthreads
                          << " threads into \"" << file_name_ << "\"";

    for (int i = 0; i < nthreads; ++i)
    {
        std::string file_name = file_name_ + std::to_string(i);
        files.push_back(TFile::Open(file_name.c_str()));
        trees.push_back((TTree*)(files[i]->Get(this->TreeName())));
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
        "sensitive_detectors", "sensitive_detectors", this->SplitLevel(), file);

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
