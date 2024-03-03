import "./css/design-area.css";

export interface HistoryDesignProps {
    imageUrl: string;
    prompt: string;
    onClick: () => void;
}

export const ForwardDesignCard: React.FC<HistoryDesignProps> = ({imageUrl, prompt, onClick}) => {    
    return (
        <div className="forward-design-card" onClick={() => onClick()}>
            <img className="forward-design-card-image" src={imageUrl}/> 
            <div className="forward-card-prompt-area"><p>{prompt}</p></div>
        </div>
    );
};