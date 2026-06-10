#include <ncl/nxsmultiformat.h>  // NCL库，用于处理系统发育数据格式
#include "json.hpp"

using json = nlohmann::json;

void processChildren(json& childrenJson, const NxsSimpleNode *parent) {
    for (auto child : parent->GetChildren()) {
        json childJson;
        if (child->GetTaxonIndex() < UINT_MAX) {  
            childJson["taxon"] = child->GetTaxonIndex();
        }
        childJson["branch_length"] = child->GetEdgeToParent().GetDblEdgeLen();
        childJson["children"] = json::array();
        processChildren(childJson["children"], child);
        childrenJson += childJson;  // 将当前子节点添加到父节点的children数组中
    }
}

int main() {
    MultiFormatReader nexusReader(-1, NxsReader::IGNORE_WARNINGS);
    nexusReader.ReadStream(std::cin, MultiFormatReader::NEXUS_FORMAT);

    json pyjson;

    pyjson["format"] = "phylojson";
    pyjson["version"] = "1.0";
    pyjson["taxa"] = json::array();
    pyjson["trees"] = json::array();

    //处理TAXA BLOCK

    const unsigned numTaxaBlocks = nexusReader.GetNumTaxaBlocks();

    if (numTaxaBlocks > 1) {
        std::cerr << "Warning: Nexus2PhyloJSON can only handle one taxa block, ignoring remaining blocks" << std::endl;
    }

    if (numTaxaBlocks > 0) {
        NxsTaxaBlock *taxaBlock = nexusReader.GetTaxaBlock(0);
        for (unsigned i = 0; i < taxaBlock->GetNTaxTotal(); i++) {
            pyjson["taxa"] += {{"id", i}, {"name", taxaBlock->GetTaxonLabel(i)}};
        }

        const unsigned numTreesBlocks = nexusReader.GetNumTreesBlocks(taxaBlock);
        if (numTreesBlocks > 1) {
            std::cerr << "Warning: Nexus2PhyloJSON can only handle one trees block, ignoring remaining blocks" << std::endl;
        }
        if (numTreesBlocks > 0) {
            NxsTreesBlock *treesBlock = nexusReader.GetTreesBlock(taxaBlock, 0);
            for (unsigned i = 0; i < treesBlock->GetNumTrees(); i++) {
                const NxsFullTreeDescription &fullTreeDesc = treesBlock->GetFullTreeDescription(i);
                NxsSimpleTree tree(fullTreeDesc, -1, -1.0);

                json treeJson;
                treeJson["name"] = fullTreeDesc.GetName();
                treeJson["root"]["branch_length"] = tree.GetRootConst()->GetEdgeToParent().GetDblEdgeLen();
                treeJson["root"]["children"] = json::array();
                processChildren(treeJson["root"]["children"], tree.GetRootConst());
                pyjson["trees"] += treeJson;
            }
        }
    }

    std::cout << pyjson.dump(4) << std::endl;
}
